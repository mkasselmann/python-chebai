import unittest
from typing import List
from unittest.mock import mock_open, patch

from chebai.preprocessing.reader import ChemAPEReader
from chebai.preprocessing.smiles_tokenizer import APETokenizer


class TestAPETokenizer(unittest.TestCase):
    """
    Unit tests for the APETokenizer class, focusing on greedy longest-match encoding.

    """

    @classmethod
    @patch(
        "chebai.preprocessing.smiles_tokenizer.open",
        new_callable=mock_open,
        read_data='{"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "C": 5, "CC": 6, "O": 7}',
    )
    def setUpClass(cls, mock_file: mock_open) -> None:
        """Set up a tokenizer with a small, deterministic (mocked) vocabulary."""
        cls.tokenizer = APETokenizer("/mock/path/vocab.json")

    def test_prefers_longest_match(self) -> None:
        """The longest vocabulary substring starting at each position must be preferred."""
        # "CC" is in the vocabulary as a single token, so it must not be split into ['C', 'C'].
        self.assertEqual(self.tokenizer.tokenize("CC"), ["CC"])

    def test_falls_back_to_single_char_match(self) -> None:
        """If no multi-character match exists, single characters are matched individually."""
        self.assertEqual(self.tokenizer.tokenize("CO"), ["C", "O"])

    def test_unknown_substrings_map_to_unk(self) -> None:
        """Characters absent from the vocabulary are replaced with the unk token."""
        self.assertEqual(self.tokenizer.tokenize("CN"), ["C", "<unk>"])

    def test_encode_returns_vocabulary_ids(self) -> None:
        """encode() must map tokens to their vocabulary indices, unk to the unk id."""
        self.assertEqual(self.tokenizer.encode("CCN"), [6, 3])


class TestChemAPEReader(unittest.TestCase):
    """
    Unit tests for the ChemAPEReader class.

    """

    @classmethod
    @patch(
        "chebai.preprocessing.smiles_tokenizer.open",
        new_callable=mock_open,
        read_data='{"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "C": 5, "CC": 6, "O": 7}',
    )
    def setUpClass(cls, mock_file: mock_open) -> None:
        """Set up a ChemAPEReader with a small, deterministic (mocked) vocabulary."""
        cls.reader = ChemAPEReader(data_path="/mock/path")

    def test_read_data_uses_fixed_vocabulary(self) -> None:
        """Encoding ethane ('CC') must use the fixed vocabulary id for the merged 'CC' token."""
        result: List[int] = self.reader._read_data("CC")
        self.assertEqual(result, [6])

    def test_read_data_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES strings must not raise and instead return None."""
        self.assertIsNone(self.reader._read_data("not_a_smiles("))

    def test_vocabulary_does_not_grow(self) -> None:
        """ChemAPEReader must never add new tokens to its (fixed, pretrained) vocabulary."""
        initial_vocab_size = len(self.reader.tokenizer.vocabulary)
        self.reader._read_data("CC(=O)O")
        self.assertEqual(len(self.reader.tokenizer.vocabulary), initial_vocab_size)


if __name__ == "__main__":
    unittest.main()
