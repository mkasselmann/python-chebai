import os
import pickle
import sys
import tempfile
import unittest
from typing import List
from unittest.mock import mock_open, patch

from chebai.preprocessing import trie_tokenizer
from chebai.preprocessing.reader import (
    EMBEDDING_OFFSET,
    TrieReader,
    TrieTTGChEMBLReader,
    TrieTTGReader,
)
from chebai.preprocessing.trie_tokenizer import (
    ReplaceTrie,
    _State,
    compress,
    tokenize,
)


def _insert_replace(root: ReplaceTrie, pattern, new_token: str) -> None:
    """Mirror of the original `trie_funcs.insert_replace` helper."""
    node = root
    for tok in pattern:
        node = node.children.setdefault(tok, ReplaceTrie())
    node.replacement = new_token


def _pickle_as_trie_funcs(obj) -> bytes:
    """Pickle `obj` (and nested ReplaceTrie/_State instances) as if it came from
    a "trie_funcs" module, matching the original SMILES-Tokenization pickle layout."""
    orig_replace_trie_module = ReplaceTrie.__module__
    orig_state_module = _State.__module__
    had_fake_module = "trie_funcs" in sys.modules
    orig_fake_module = sys.modules.get("trie_funcs")
    try:
        ReplaceTrie.__module__ = "trie_funcs"
        _State.__module__ = "trie_funcs"
        # pickle verifies the class is importable from its recorded module.
        sys.modules["trie_funcs"] = trie_tokenizer
        return pickle.dumps(obj)
    finally:
        ReplaceTrie.__module__ = orig_replace_trie_module
        _State.__module__ = orig_state_module
        if had_fake_module:
            sys.modules["trie_funcs"] = orig_fake_module
        else:
            del sys.modules["trie_funcs"]


class TestTrieFunctions(unittest.TestCase):
    """
    Unit tests for the module-level `tokenize`/`compress` functions.

    """

    def test_tokenize_splits_atoms(self) -> None:
        """The trie's regex must split a simple SMILES into atom-level tokens."""
        self.assertEqual(tokenize("CC(=O)O"), ["C", "C", "(", "=", "O", ")", "O"])

    def test_compress_replaces_longest_match(self) -> None:
        """compress() must greedily replace the longest matching token run."""
        root = ReplaceTrie()
        _insert_replace(root, ("C", "C"), "<R0>")
        _insert_replace(root, ("C", "C", "O"), "<R1>")
        # "CCO" -> longest match is ('C','C','O') -> single replacement token
        self.assertEqual(compress(tokenize("CCO"), root), ["<R1>"])
        # "CCN" -> only ('C','C') matches -> ['<R0>', 'N']
        self.assertEqual(compress(tokenize("CCN"), root), ["<R0>", "N"])

    def test_compress_leaves_unmatched_tokens(self) -> None:
        """Tokens with no matching pattern in the trie are returned unchanged."""
        root = ReplaceTrie()
        _insert_replace(root, ("C", "C"), "<R0>")
        self.assertEqual(compress(tokenize("NO"), root), ["N", "O"])


class TestTrieReader(unittest.TestCase):
    """
    Unit tests for the TrieReader class, including safe unpickling of a pretrained trie.

    """

    @classmethod
    def setUpClass(cls) -> None:
        """Build a tiny pretrained trie (pickled as if from "trie_funcs") and load it."""
        root = ReplaceTrie()
        _insert_replace(root, ("C", "C"), "<R0>")
        state = _State(token_to_idx={}, idx_to_token={}, replace_root=root)

        trie_fd, cls.trie_path = tempfile.mkstemp(suffix=".pkl")
        with os.fdopen(trie_fd, "wb") as f:
            f.write(_pickle_as_trie_funcs(state))

        with patch(
            "chebai.preprocessing.reader.open",
            new_callable=mock_open,
            read_data="",
        ):
            cls.reader = TrieReader(token_path="/mock/path", trie_path=cls.trie_path)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.trie_path)

    def test_read_data_adds_new_tokens(self) -> None:
        """New (replaced or raw) tokens are assigned increasing indices and cached."""
        # canonical SMILES for ethane is "CC" -> replaced with the pretrained "<R0>" token,
        # but the cache must store the resolved literal substring, not "<R0>" itself.
        result: List[int] = self.reader._read_data("CC")
        self.assertEqual(result, [EMBEDDING_OFFSET + 0])
        self.assertIn("CC", self.reader.cache)
        self.assertNotIn("<R0>", self.reader.cache)

    def test_read_data_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES strings must not raise and instead return None."""
        self.assertIsNone(self.reader._read_data("not_a_smiles("))


class TestTrieTTGReader(unittest.TestCase):
    """
    Unit tests for the TrieTTGReader class (same pickle layout as TrieReader, distinct token cache).

    """

    @classmethod
    def setUpClass(cls) -> None:
        """Build a tiny pretrained trie (pickled as if from "trie_funcs") and load it."""
        root = ReplaceTrie()
        _insert_replace(root, ("C", "C"), "<R0>")
        state = _State(token_to_idx={}, idx_to_token={}, replace_root=root)

        trie_fd, cls.trie_path = tempfile.mkstemp(suffix=".pkl")
        with os.fdopen(trie_fd, "wb") as f:
            f.write(_pickle_as_trie_funcs(state))

        with patch(
            "chebai.preprocessing.reader.open",
            new_callable=mock_open,
            read_data="",
        ):
            cls.reader = TrieTTGReader(token_path="/mock/path", trie_path=cls.trie_path)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.trie_path)

    def test_name_is_distinct_from_trie_reader(self) -> None:
        """TrieTTGReader must use its own token cache name, not collide with TrieReader."""
        self.assertEqual(TrieTTGReader.name(), "smiles_ttg")
        self.assertNotEqual(TrieTTGReader.name(), TrieReader.name())

    def test_read_data_adds_new_tokens(self) -> None:
        """New (replaced or raw) tokens are assigned increasing indices and cached."""
        result: List[int] = self.reader._read_data("CC")
        self.assertEqual(result, [EMBEDDING_OFFSET + 0])
        self.assertIn("CC", self.reader.cache)
        self.assertNotIn("<R0>", self.reader.cache)

    def test_read_data_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES strings must not raise and instead return None."""
        self.assertIsNone(self.reader._read_data("not_a_smiles("))


class TestTrieTTGChEMBLReader(unittest.TestCase):
    """
    Unit tests for the TrieTTGChEMBLReader class (same pickle layout, distinct token cache).

    """

    @classmethod
    def setUpClass(cls) -> None:
        """Build a tiny pretrained trie (pickled as if from "trie_funcs") and load it."""
        root = ReplaceTrie()
        _insert_replace(root, ("C", "C"), "<R0>")
        state = _State(token_to_idx={}, idx_to_token={}, replace_root=root)

        trie_fd, cls.trie_path = tempfile.mkstemp(suffix=".pkl")
        with os.fdopen(trie_fd, "wb") as f:
            f.write(_pickle_as_trie_funcs(state))

        with patch(
            "chebai.preprocessing.reader.open",
            new_callable=mock_open,
            read_data="",
        ):
            cls.reader = TrieTTGChEMBLReader(
                token_path="/mock/path", trie_path=cls.trie_path
            )

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.trie_path)

    def test_name_is_distinct_from_other_trie_readers(self) -> None:
        """TrieTTGChEMBLReader must use its own token cache name."""
        self.assertEqual(TrieTTGChEMBLReader.name(), "smiles_ttg_chembl")
        self.assertNotEqual(TrieTTGChEMBLReader.name(), TrieReader.name())
        self.assertNotEqual(TrieTTGChEMBLReader.name(), TrieTTGReader.name())

    def test_read_data_adds_new_tokens(self) -> None:
        """New (replaced or raw) tokens are assigned increasing indices and cached."""
        result: List[int] = self.reader._read_data("CC")
        self.assertEqual(result, [EMBEDDING_OFFSET + 0])
        self.assertIn("CC", self.reader.cache)
        self.assertNotIn("<R0>", self.reader.cache)

    def test_read_data_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES strings must not raise and instead return None."""
        self.assertIsNone(self.reader._read_data("not_a_smiles("))


if __name__ == "__main__":
    unittest.main()
