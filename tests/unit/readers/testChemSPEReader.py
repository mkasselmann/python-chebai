import os
import tempfile
import unittest
from typing import List
from unittest.mock import mock_open, patch

from chebai.preprocessing.reader import EMBEDDING_OFFSET, ChemSPEReader
from chebai.preprocessing.smiles_tokenizer import SPETokenizer


class TestSPETokenizer(unittest.TestCase):
    """
    Unit tests for the SPETokenizer class, focusing on the ranked, iterative merge algorithm.

    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up a tokenizer with a small, deterministic codes file (ranked merge rules)."""
        codes_fd, cls.codes_path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(codes_fd, "w") as f:
            # rank 0: highest priority merge
            f.write("c c\n")
            # rank 1
            f.write("cc cc\n")

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.codes_path)

    def test_merges_are_applied_by_rank(self) -> None:
        """Higher-ranked (earlier) merge rules must be applied before lower-ranked ones."""
        tokenizer = SPETokenizer(self.codes_path)
        # "cccc" -> ['c','c','c','c'] -[rank 0: c+c]-> ['cc','cc'] -[rank 1: cc+cc]-> ['cccc']
        self.assertEqual(tokenizer.tokenize("cccc"), ["cccc"])

    def test_merge_stops_when_no_ranked_pair_remains(self) -> None:
        """Merging must stop once no remaining adjacent pair has a merge rule."""
        tokenizer = SPETokenizer(self.codes_path)
        # "ccc" -> ['c','c','c'] -[rank 0: c+c, left-to-right, non-overlapping]-> ['cc', 'c']
        # ('cc', 'c') has no merge rule -> stop
        self.assertEqual(tokenizer.tokenize("ccc"), ["cc", "c"])

    def test_unmergeable_tokens_are_left_as_is(self) -> None:
        """Tokens with no applicable merge rule are returned unchanged, atom-by-atom."""
        tokenizer = SPETokenizer(self.codes_path)
        self.assertEqual(tokenizer.tokenize("NO"), ["N", "O"])


class TestChemSPEReader(unittest.TestCase):
    """
    Unit tests for the ChemSPEReader class.

    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up a real SPE codes file and a ChemSPEReader with a mocked (empty) token cache."""
        codes_fd, cls.codes_path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(codes_fd, "w") as f:
            f.write("C C\n")

        with patch(
            "chebai.preprocessing.reader.open",
            new_callable=mock_open,
            read_data="",
        ):
            cls.reader = ChemSPEReader(
                token_path="/mock/path", codes_path=cls.codes_path
            )

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.codes_path)

    def test_read_data_adds_new_tokens(self) -> None:
        """New (merged) tokens are assigned increasing indices and added to the cache."""
        # canonical SMILES for ethane is "CC" -> merges (rank 0: C+C) into a single "CC" token
        result: List[int] = self.reader._read_data("CC")
        self.assertEqual(result, [EMBEDDING_OFFSET + 0])
        self.assertIn("CC", self.reader.cache)

    def test_read_data_reuses_cached_tokens(self) -> None:
        """Encoding the same SMILES twice must not grow the cache further."""
        self.reader._read_data("CC")
        cache_size_before = len(self.reader.cache)
        self.reader._read_data("CC")
        self.assertEqual(len(self.reader.cache), cache_size_before)

    def test_read_data_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES strings must not raise and instead return None."""
        self.assertIsNone(self.reader._read_data("not_a_smiles("))


if __name__ == "__main__":
    unittest.main()
