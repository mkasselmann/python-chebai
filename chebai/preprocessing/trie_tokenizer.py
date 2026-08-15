"""
Loader/inference code for the trie-based SMILES compressor from the
SMILES-Tokenization project (https://github.com/BlastCoder/SMILES-Tokenization).

The pretrained trie is a pickled `_State` namedtuple (produced by
`train/train_trie.py::save_state`) holding a `ReplaceTrie` built from frequent
token n-grams (Algorithms 3-7 in the accompanying paper). This module re-implements
just enough of the original `trie_funcs.py` (same regex, same node/trie layout) to:
  1. Safely unpickle the bundled ``trie_pubchem100K.pkl`` (via an allow-listed
     `pickle.Unpickler` that only reconstructs `_State`/`ReplaceTrie`, since the
     file was pickled under the module name "trie_funcs" which doesn't exist here).
  2. Tokenize new SMILES with the same greedy longest-match compression used at
     training time.
"""

from __future__ import annotations

import collections
import pickle
import re
from typing import Dict, List, Optional

# Same pretokenization regex as trie_funcs.py, required to match the trie's training data.
TOKEN_PATTERN = re.compile(
    r"(\[[^\[\]]+\]|Br?|Cl?|[A-Z][a-z]?|\d+|=|\/|\\|\+|\-|\(|\)|@|\[|\])"
)


def tokenize(smiles: str) -> List[str]:
    """Split a SMILES string into atom-level tokens (matches trie training)."""
    return TOKEN_PATTERN.findall(smiles)


class ReplaceTrie:
    """Trie node used for greedy longest-match token-sequence replacement."""

    __slots__ = ("children", "replacement")

    def __init__(self) -> None:
        self.children: Dict[str, "ReplaceTrie"] = {}
        self.replacement: Optional[str] = None


class _State(
    collections.namedtuple("_State", "token_to_idx idx_to_token replace_root")
):
    """Mirrors the pickled namedtuple layout from the original `trie_funcs.py`."""

    __slots__ = ()


def compress(tokens: List[str], rt_root: ReplaceTrie) -> List[str]:
    """Greedily replace the longest matching token run using `rt_root` (Algorithm 7)."""
    out: List[str] = []
    i, n = 0, len(tokens)
    while i < n:
        node = rt_root
        j = i
        last_rep: Optional[str] = None
        last_len = 0
        while j < n and tokens[j] in node.children:
            node = node.children[tokens[j]]
            if node.replacement is not None:
                last_rep, last_len = node.replacement, j - i + 1
            j += 1
        if last_rep is not None:
            out.append(last_rep)
            i += last_len
        else:
            out.append(tokens[i])
            i += 1
    return out


class _RestrictedTrieUnpickler(pickle.Unpickler):
    """
    Only reconstructs the exact classes needed for a trie `_State`; refuses everything
    else. The original pickle references classes under the module name "trie_funcs"
    (the training script), which is redirected here to our reimplementation.
    """

    _ALLOWED = {
        ("trie_funcs", "_State"): _State,
        ("trie_funcs", "ReplaceTrie"): ReplaceTrie,
    }

    def find_class(self, module: str, name: str):
        try:
            return self._ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle disallowed class '{module}.{name}'"
            )


class TrieTokenizer:
    """
    Longest-match SMILES tokenizer using a pretrained replacement trie.

    Args:
        trie_path: Path to the pickled trie `_State` (as produced by
            `train_trie.py` from https://github.com/BlastCoder/SMILES-Tokenization).
    """

    def __init__(self, trie_path: str):
        with open(trie_path, "rb") as f:
            state: _State = _RestrictedTrieUnpickler(f).load()
        self.replace_root = state.replace_root

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string, replacing frequent token runs via the trie."""
        return compress(tokenize(smiles), self.replace_root)
