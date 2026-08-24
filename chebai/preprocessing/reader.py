import inspect
import os
import sys
from abc import ABC
from itertools import islice
from typing import Any, Dict, List, Optional

from pysmiles.read_smiles import _tokenize
from rdkit import Chem

from chebai.preprocessing.collate import DefaultCollator, RaggedCollator

EMBEDDING_OFFSET = 10
PADDING_TOKEN_INDEX = 0
MASK_TOKEN_INDEX = 1
CLS_TOKEN = 2


class DataReader:
    """
    Base class for reading and preprocessing data. Turns the raw input data (e.g., a SMILES string) into the model
    input format (e.g., a list of tokens).

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments (not used).
    """

    COLLATOR = DefaultCollator

    def __init__(
        self,
        collator_kwargs: Optional[Dict[str, Any]] = None,
        token_path: Optional[str] = None,
        **kwargs,
    ):
        if collator_kwargs is None:
            collator_kwargs = dict()
        self.collator = self.COLLATOR(**collator_kwargs)
        self.dirname = os.path.dirname(inspect.getfile(self.__class__))
        self._token_path = token_path

    def _get_raw_data(self, row: Dict[str, Any]) -> Any:
        """Get raw data from the row."""
        return row["features"]

    def _get_raw_label(self, row: Dict[str, Any]) -> Any:
        """Get raw label from the row."""
        return row["labels"]

    def _get_raw_id(self, row: Dict[str, Any]) -> Any:
        """Get raw ID from the row."""
        return row.get("ident", row["features"])

    def _get_raw_group(self, row: Dict[str, Any]) -> Any:
        """Get raw group from the row."""
        return row.get("group", None)

    def _get_additional_kwargs(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional keyword arguments from the row."""
        return row.get("additional_kwargs", dict())

    def name(cls) -> str:
        """Returns the name of the data reader."""
        raise NotImplementedError

    @property
    def token_path(self) -> str:
        """Get token path, create file if it does not exist yet."""
        if self._token_path is not None:
            return self._token_path
        token_path = os.path.join(self.dirname, "bin", self.name(), "tokens.txt")
        os.makedirs(os.path.join(self.dirname, "bin", self.name()), exist_ok=True)
        if not os.path.exists(token_path):
            with open(token_path, "x"):
                pass
        return token_path

    def _read_id(self, raw_data: Any) -> Any:
        """Read and return ID from raw data."""
        return raw_data

    def _read_data(self, raw_data: Any) -> Any:
        """Read and return data from raw data."""
        return raw_data

    def _read_label(self, raw_label: Any) -> Any:
        """Read and return label from raw label."""
        return raw_label

    def _read_group(self, raw: Any) -> Any:
        """Read and return group from raw group data."""
        return raw

    def _read_components(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Read and return components from the row. If the data contains any missing labels (`None`), they are tracked
        under the additional `missing_labels` keyword."""
        labels = self._get_raw_label(row)
        additional_kwargs = self._get_additional_kwargs(row)
        if labels is not None:
            if any(label is None for label in labels):
                additional_kwargs["missing_labels"] = [
                    label is None for label in labels
                ]
        return dict(
            features=self._get_raw_data(row),
            labels=labels,
            ident=self._get_raw_id(row),
            group=self._get_raw_group(row),
            additional_kwargs=additional_kwargs,
        )

    def to_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw row data to processed data."""
        d = self._read_components(row)
        return dict(
            features=self._read_data(d["features"]),
            labels=self._read_label(d["labels"]),
            ident=self._read_id(d["ident"]),
            group=self._read_group(d["group"]),
            **d["additional_kwargs"],
        )

    def on_finish(self) -> None:
        """Hook to run at the end of preprocessing."""
        return


class TokenIndexerReader(DataReader, ABC):
    """
    Abstract base class for reading tokenized data and mapping tokens to unique indices.

    This class maintains a cache of token-to-index mappings that can be extended during runtime,
    and saves new tokens to a persistent file at the end of processing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.token_path, "r") as pk:
            self.cache: Dict[str, int] = {
                token.strip(): idx for idx, token in enumerate(pk)
            }
        self._loaded_tokens_count = len(self.cache)

    def _get_token_index(self, token: str) -> int:
        """Returns a unique number for each token, automatically adds new tokens."""
        if str(token) not in self.cache:
            self.cache[(str(token))] = len(self.cache)
        return self.cache[str(token)] + EMBEDDING_OFFSET

    def on_finish(self) -> None:
        """
        Saves the current cache of tokens to the token file.This method is called after all data processing is complete.
        """
        print(f"first 10 tokens: {list(islice(self.cache, 10))}")

        total_tokens = len(self.cache)
        if total_tokens > self._loaded_tokens_count:
            print("New tokens added to the cache, Saving them to token file.....")

            assert sys.version_info >= (
                3,
                7,
            ), "This code requires Python 3.7 or higher."
            # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
            # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
            # https://mail.python.org/pipermail/python-dev/2017-December/151283.html
            new_tokens = list(
                islice(self.cache, self._loaded_tokens_count, total_tokens)
            )

            with open(self.token_path, "a") as pk:
                print(f"saving new {len(new_tokens)} tokens to {self.token_path}...")
                pk.writelines([f"{c}\n" for c in new_tokens])


class ChemDataReader(TokenIndexerReader):
    """
    Data reader for chemical data using SMILES tokens.
    """

    COLLATOR = RaggedCollator

    def __init__(self, canonicalize_smiles=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.canonicalize_smiles = canonicalize_smiles
        print(f"Using SMILES canonicalization: {self.canonicalize_smiles}")

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_token"

    def _read_data(self, raw_data: str | Chem.Mol) -> Optional[List[int]]:
        """
        Reads and tokenizes SMILES strings (or SMILES strings generated from Chem.Mol objects) into a list of token indices. Optionally canonicalizes the SMILES string using RDKit.

        Args:
            raw_data (str|Chem.Mol): The raw SMILES string or Chem.Mol object to be tokenized.

        Returns:
            List[int]: A list of integers representing the indices of the SMILES tokens.
        """
        try:
            if isinstance(raw_data, str):
                mol = Chem.MolFromSmiles(raw_data.strip())
            else:
                mol = raw_data
            if mol is None:
                raise ValueError(f"Invalid input: {raw_data}")
        except ValueError as e:
            print(f"Could not process {raw_data}")
            print(f"\tError: {e}")
            return None

        if self.canonicalize_smiles:
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                print(f"RDKit failed to canonicalize the SMILES: {raw_data}")
                print(f"\t{e}")
                return None
        elif not isinstance(raw_data, str):
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception as e:
                print(f"RDKit failed to convert Mol object to SMILES: {raw_data}")
                print(f"\t{e}")
                return None
        else:
            smiles = raw_data

        try:
            tokenized = [self._get_token_index(v[1]) for v in _tokenize(smiles)]
        except ValueError as e:
            print(f"Could not tokenize SMILES: {smiles}")
            print(f"\tError: {e}")
            return None
        return tokenized

    def _back_to_smiles(self, smiles_encoded):
        token_file = self.reader.token_path
        token_coding = {}
        counter = 0
        smiles_decoded = ""

        # todo: for now just copied over from a notebook but ideally do this using the cache
        with open(token_file, "r") as file:
            for line in file:
                token_coding[counter] = line.strip()
                counter += 1

        for token in smiles_encoded:
            smiles_decoded += token_coding[token - EMBEDDING_OFFSET]

        return smiles_decoded


class StaticSMILESReader(DataReader):
    """
    Data reader for SMILES tokens with a static token set. Atoms are split into 5 components: isotope, element, charge, hydrogens, stereo.
    New tokens are not added to the token file, and unknown tokens are mapped to a special index.
    """

    COLLATOR = RaggedCollator

    def __init__(self, *args, **kwargs) -> None:
        from chebai.preprocessing.smiles_tokenizer import BasicSmilesTokenizer

        super().__init__(*args, **kwargs)
        self.tokenizer = BasicSmilesTokenizer()

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "static_smiles"

    def _read_data(self, raw_data: str | Chem.Mol) -> Optional[List[int]]:
        """Tokenize raw SMILES data using BasicSmilesTokenizer with static vocabulary."""
        try:
            if isinstance(raw_data, str):
                mol = Chem.MolFromSmiles(raw_data.strip())
            else:
                mol = raw_data
        except ValueError as e:
            print(f"could not process {raw_data}")
            print(f"\tError: {e}")
            return None

        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            print(f"RDKit failed to canonicalize the SMILES: {raw_data}")
            print(f"\t{e}")
            return None

        try:
            return self.tokenizer.encode(smiles)
        except Exception as e:
            print(f"could not tokenize {raw_data}")
            print(f"\tError: {e}")
            return None


class DeepChemDataReader(ChemDataReader):
    """
    Data reader for chemical data using DeepSMILES tokens.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        import deepsmiles

        super().__init__(*args, **kwargs)
        self.converter = deepsmiles.Converter(rings=True, branches=True)
        self.error_count = 0

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "deepsmiles_token"

    def _read_data(self, raw_data: str) -> Optional[List[int]]:
        """Read and tokenize raw data using DeepSMILES."""
        try:
            tokenized = _tokenize(self.converter.encode(raw_data))
            tokenized = [self._get_token_index(v[1]) for v in tokenized]
        except ValueError as e:
            print(f"could not process {raw_data}")
            print(f"Corresponding deepSMILES: {self.converter.encode(raw_data)}")
            print(f"\t{e}")
            self.error_count += 1
            print(f"\terror count: {self.error_count}")
            tokenized = None
        return tokenized


class ChemDataUnlabeledReader(ChemDataReader):
    """
    Data reader for unlabeled chemical data using SMILES tokens.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_token_unlabeled"

    def _get_raw_label(self, row: Dict[str, Any]) -> None:
        """Returns None as there are no labels."""
        return None


class ChemBPEReader(DataReader):
    """
    Data reader for chemical data using BPE tokenization.

    Args:
        data_path: Path to a directory holding a pretrained (HuggingFace-style) BPE tokenizer,
            i.e. a "vocab.json"/"merges.txt" pair (optionally with "tokenizer_config.json" etc.).
            Defaults to the bundled tokenizer under "bin/BPE_SWJ".
        max_len: Maximum length of the tokenized sequence.
        vsize: Vocabulary size for the tokenizer (not used).
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_bpe"

    def __init__(
        self,
        *args,
        data_path: Optional[str] = None,
        max_len: int = 1800,
        vsize: int = 4000,
        **kwargs,
    ):
        from tokenizers.implementations import ByteLevelBPETokenizer
        from tokenizers.processors import RobertaProcessing

        super().__init__(*args, **kwargs)
        if data_path is None:
            data_path = os.path.join(self.dirname, "bin", "BPE_SWJ")
        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(data_path, "vocab.json"),
            os.path.join(data_path, "merges.txt"),
        )
        self.tokenizer.post_processor = RobertaProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=max_len)

    def _get_raw_data(self, row: Dict[str, Any]) -> List[int]:
        """Tokenize raw data using BPE tokenizer."""
        return self.tokenizer.encode(row["features"]).ids


class ChemSPEReader(TokenIndexerReader):
    """
    Data reader for chemical data using SMILES Pair Encoding (SPE) tokenization.

    Applies a pretrained SPE codes file (ranked atom-pair merge rules, e.g. produced by
    https://github.com/XinhaoLi74/SmilesPE) to tokenize SMILES strings. Unlike
    `ChemBPEReader`, the vocabulary (token -> index mapping) is not fixed; new tokens
    encountered at runtime are added to the cache, like `ChemDataReader`.

    Args:
        codes_path: Path to the SPE codes file (ranked merge rules). Defaults to the
            bundled "bin/spe_pubchem100K/spe_pubchem100K.txt".
        canonicalize_smiles: Whether to canonicalize SMILES using RDKit before tokenizing.
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    def __init__(
        self,
        *args,
        codes_path: Optional[str] = None,
        canonicalize_smiles: bool = True,
        **kwargs,
    ) -> None:
        from chebai.preprocessing.smiles_tokenizer import SPETokenizer

        super().__init__(*args, **kwargs)
        if codes_path is None:
            codes_path = os.path.join(
                self.dirname, "bin", "spe_pubchem100K", "spe_pubchem100K.txt"
            )
        self.tokenizer = SPETokenizer(codes_path)
        self.canonicalize_smiles = canonicalize_smiles

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_spe"

    def _read_data(self, raw_data: str | Chem.Mol) -> Optional[List[int]]:
        """Tokenize a SMILES string (or Chem.Mol) into a list of SPE token indices."""
        try:
            if isinstance(raw_data, str):
                mol = Chem.MolFromSmiles(raw_data.strip())
            else:
                mol = raw_data
            if mol is None:
                raise ValueError(f"Invalid input: {raw_data}")
        except ValueError as e:
            print(f"Could not process {raw_data}")
            print(f"\tError: {e}")
            return None

        if self.canonicalize_smiles:
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                print(f"RDKit failed to canonicalize the SMILES: {raw_data}")
                print(f"\t{e}")
                return None
        elif isinstance(raw_data, str):
            smiles = raw_data
        else:
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception as e:
                print(f"RDKit failed to convert Mol object to SMILES: {raw_data}")
                print(f"\t{e}")
                return None

        try:
            tokenized = [
                self._get_token_index(tok) for tok in self.tokenizer.tokenize(smiles)
            ]
        except Exception as e:
            print(f"Could not tokenize SMILES: {smiles}")
            print(f"\tError: {e}")
            return None
        return tokenized


class SPEChEMBLReader(ChemSPEReader):
    DEFAULT_TRIE_SUBDIR = "spe_chembl"
    DEFAULT_TRIE_FILENAME = "spe_chembl.txt"

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_spe_chembl"


class ChemAPEReader(DataReader):
    """
    Data reader for chemical data using a pretrained Atom Pair Encoding (APE) tokenizer,
    from the SMILES-Tokenization project (https://github.com/BlastCoder/SMILES-Tokenization).

    Uses a fixed pretrained vocabulary with greedy longest-match tokenization; unlike `ChemSPEReader`/`TrieReader`,
    unmatched substrings are mapped to a fixed unknown-token id rather than being added
    to the vocabulary.

    Args:
        data_path: Path to a directory holding a pretrained APE tokenizer, i.e. a
            "vocab.json" file. Defaults to the bundled tokenizer under "bin/ape_pubchem100K".
        canonicalize_smiles: Whether to canonicalize SMILES using RDKit before tokenizing.
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_ape"

    def __init__(
        self,
        *args,
        data_path: Optional[str] = None,
        canonicalize_smiles: bool = True,
        **kwargs,
    ):
        from chebai.preprocessing.smiles_tokenizer import APETokenizer

        super().__init__(*args, **kwargs)
        if data_path is None:
            data_path = os.path.join(self.dirname, "bin", "ape_pubchem100K")
        self.tokenizer = APETokenizer(os.path.join(data_path, "vocab.json"))
        self.canonicalize_smiles = canonicalize_smiles

    def _read_data(self, raw_data: str | Chem.Mol) -> Optional[List[int]]:
        """Tokenize a SMILES string (or Chem.Mol) using the pretrained APE vocabulary."""
        try:
            if isinstance(raw_data, str):
                mol = Chem.MolFromSmiles(raw_data.strip())
            else:
                mol = raw_data
            if mol is None:
                raise ValueError(f"Invalid input: {raw_data}")
        except ValueError as e:
            print(f"Could not process {raw_data}")
            print(f"\tError: {e}")
            return None

        if self.canonicalize_smiles:
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                print(f"RDKit failed to canonicalize the SMILES: {raw_data}")
                print(f"\t{e}")
                return None
        elif isinstance(raw_data, str):
            smiles = raw_data
        else:
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception as e:
                print(f"RDKit failed to convert Mol object to SMILES: {raw_data}")
                print(f"\t{e}")
                return None

        try:
            return self.tokenizer.encode(smiles)
        except Exception as e:
            print(f"Could not tokenize SMILES: {smiles}")
            print(f"\tError: {e}")
            return None


class APEChEMBLReader(ChemAPEReader):
    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_ape_chembl"


class TrieReader(TokenIndexerReader):
    """
    Data reader for chemical data using a pretrained replacement-trie tokenizer,
    from the SMILES-Tokenization project (https://github.com/BlastCoder/SMILES-Tokenization).

    Like `ChemSPEReader`, the vocabulary is not fixed: new (replacement or raw) tokens
    encountered at runtime are added to the cache.

    Args:
        trie_path: Path to the pickled trie `_State`. Defaults to the bundled
            "bin/trie_pubchem100K/trie_pubchem100K.pkl".
        canonicalize_smiles: Whether to canonicalize SMILES using RDKit before tokenizing.
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator
    DEFAULT_TRIE_SUBDIR = "trie_pubchem100K"
    DEFAULT_TRIE_FILENAME = "trie_pubchem100K.pkl"

    def __init__(
        self,
        *args,
        trie_path: Optional[str] = None,
        canonicalize_smiles: bool = True,
        **kwargs,
    ) -> None:
        from chebai.preprocessing.trie_tokenizer import TrieTokenizer

        super().__init__(*args, **kwargs)
        if trie_path is None:
            trie_path = os.path.join(
                self.dirname,
                "bin",
                self.DEFAULT_TRIE_SUBDIR,
                self.DEFAULT_TRIE_FILENAME,
            )
        self.tokenizer = TrieTokenizer(trie_path)
        self.canonicalize_smiles = canonicalize_smiles

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_trie"

    def _read_data(self, raw_data: str | Chem.Mol) -> Optional[List[int]]:
        """Tokenize a SMILES string (or Chem.Mol) into a list of trie token indices."""
        try:
            if isinstance(raw_data, str):
                mol = Chem.MolFromSmiles(raw_data.strip())
            else:
                mol = raw_data
            if mol is None:
                raise ValueError(f"Invalid input: {raw_data}")
        except ValueError as e:
            print(f"Could not process {raw_data}")
            print(f"\tError: {e}")
            return None

        if self.canonicalize_smiles:
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                print(f"RDKit failed to canonicalize the SMILES: {raw_data}")
                print(f"\t{e}")
                return None
        elif isinstance(raw_data, str):
            smiles = raw_data
        else:
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception as e:
                print(f"RDKit failed to convert Mol object to SMILES: {raw_data}")
                print(f"\t{e}")
                return None

        try:
            # Resolve each "<R#>" replacement token back to the literal substring it
            # stands for, so the token cache/tokens.txt stores real SMILES fragments
            # instead of the artificial trie replacement tokens.
            tokenized = [
                self._get_token_index("".join(self.tokenizer.detokenize([tok])))
                for tok in self.tokenizer.tokenize(smiles)
            ]
        except Exception as e:
            print(f"Could not tokenize SMILES: {smiles}")
            print(f"\tError: {e}")
            return None
        return tokenized


class TrieChEMBLReader(TrieReader):
    DEFAULT_TRIE_SUBDIR = "trie_chembl"
    DEFAULT_TRIE_FILENAME = "trie_chembl.pkl"

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_trie_chembl"


class TrieChEBIReader(TrieReader):
    DEFAULT_TRIE_SUBDIR = "trie_chebi"
    DEFAULT_TRIE_FILENAME = "trie_chebi.pkl"

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_trie_chebi"


class TrieTTGReader(TrieReader):
    """
    Data reader using the TTG (Token Transition Graph) refined replacement trie
    from the SMILES-Tokenization project (https://github.com/BlastCoder/SMILES-Tokenization).

    Structurally identical to `TrieReader` (same pickled `_State`/`ReplaceTrie` layout),
    but the trie was built with entropy-based filtering (Algorithm 8): patterns are only
    kept if their average token-transition entropy stays below a threshold, which changes
    which n-grams get merged. Uses its own token cache, separate from `TrieReader`.

    Args:
        trie_path: Path to the pickled trie `_State`. Defaults to the bundled
            "bin/ttg_pubchem100K/ttg_pubchem100K_K8_F3_H3p5.pkl".
        canonicalize_smiles: Whether to canonicalize SMILES using RDKit before tokenizing.
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    DEFAULT_TRIE_SUBDIR = "ttg_pubchem100K"
    DEFAULT_TRIE_FILENAME = "ttg_pubchem100K_K8_F3_H3p5.pkl"

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_ttg"


class TrieTTGChEMBLReader(TrieReader):
    """
    Data reader using the TTG-refined replacement trie trained on ChEMBL, from the
    SMILES-Tokenization project (https://github.com/BlastCoder/SMILES-Tokenization).

    Structurally identical to `TrieReader`/`TrieTTGReader` (same pickled `_State`/
    `ReplaceTrie` layout), only the underlying training corpus differs (ChEMBL instead
    of PubChem). Uses its own token cache, separate from the other Trie-based readers.

    Args:
        trie_path: Path to the pickled trie `_State`. Defaults to the bundled
            "bin/ttg_chembl/trie_ttg_chembl.pkl".
        canonicalize_smiles: Whether to canonicalize SMILES using RDKit before tokenizing.
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    DEFAULT_TRIE_SUBDIR = "ttg_chembl"
    DEFAULT_TRIE_FILENAME = "trie_ttg_chembl.pkl"

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_ttg_chembl"


class TrieTTGChEBIReader(TrieReader):
    """
    Data reader using the TTG-refined replacement trie trained on ChEBI, from the
    SMILES-Tokenization project (https://github.com/BlastCoder/SMILES-Tokenization).

    Structurally identical to `TrieReader`/`TrieTTGReader` (same pickled `_State`/
    `ReplaceTrie` layout), only the underlying training corpus differs (ChEBI instead
    of PubChem). Uses its own token cache, separate from the other Trie-based readers.

    Args:
        trie_path: Path to the pickled trie `_State`. Defaults to the bundled
            "bin/ttg_chebi/ttg_chebi_K8_F4_H2p0.pkl".
        canonicalize_smiles: Whether to canonicalize SMILES using RDKit before tokenizing.
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    DEFAULT_TRIE_SUBDIR = "ttg_chebi"
    DEFAULT_TRIE_FILENAME = "ttg_chebi_K8_F4_H2p0.pkl"

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "smiles_ttg_chebi"


class SelfiesReader(ChemDataReader):
    """
    Data reader for chemical data using SELFIES tokens.

    Args:
        data_path: Path for the pretrained BPE tokenizer.
        max_len: Maximum length of the tokenized sequence.
        vsize: Vocabulary size for the tokenizer.
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    def __init__(
        self,
        *args,
        data_path: Optional[str] = None,
        max_len: int = 1800,
        vsize: int = 4000,
        **kwargs,
    ):
        import selfies as sf

        super().__init__(*args, **kwargs)
        self.error_count = 0
        sf.set_semantic_constraints("hypervalent")

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "selfies"

    def _read_data(self, raw_data: str) -> Optional[List[int]]:
        """Read and tokenize raw data using SELFIES."""
        import selfies as sf

        try:
            tokenized = sf.split_selfies(sf.encoder(raw_data.strip(), strict=True))
            tokenized = [self._get_token_index(v) for v in tokenized]
        except Exception:
            print(f"could not process {raw_data}")
            # print(f'\t{e}')
            self.error_count += 1
            print(f"\terror count: {self.error_count}")
            tokenized = None
            # if self.error_count > 20:
            #    raise Exception('Too many errors')
        return tokenized


class OrdReader(DataReader):
    """
    Data reader that converts characters to their ordinal values.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        token_path: Optional path for the token file.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = RaggedCollator

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "ord"

    def _read_data(self, raw_data: str) -> List[int]:
        """Convert characters in raw data to their ordinal values."""
        return [ord(s) for s in raw_data]


class FingerprintReader(DataReader):
    """
    Data reader for chemical data using RDKit fingerprints.

    Args:
        collator_kwargs: Optional dictionary of keyword arguments for the collator.
        kwargs: Additional keyword arguments.
    """

    COLLATOR = DefaultCollator

    def __init__(self, fingerprint_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fingerprint_size = fingerprint_size

    @classmethod
    def name(cls) -> str:
        """Returns the name of the data reader."""
        return "rdkit_fingerprint"

    def _read_data(self, raw_data: str) -> List[int]:
        """Generate RDKit fingerprint from raw SMILES data."""
        mol = Chem.MolFromSmiles(raw_data.strip())
        if mol is None:
            raise ValueError(f"Invalid SMILES: {raw_data}")
        return list(
            Chem.RDKFingerprint(mol, fpSize=self.fingerprint_size).ToBitString()
        )
