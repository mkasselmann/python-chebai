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
import os
import pickle
import re
from typing import Dict, List, Optional


TOKEN_PATTERN = re.compile(
    r"(\[[^\[\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|%\d{2}|\d+|=|#|:|\/|\\|\+|\-|\(|\)|@@|@|\[|\])"
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


def build_reverse_map(rt_root: ReplaceTrie) -> Dict[str, List[str]]:
    """Walk `rt_root` and map every replacement token (e.g. "<R0>") to the original
    token sequence it was substituted for (inverse of `compress`)."""
    reverse_map: Dict[str, List[str]] = {}
    stack: List[tuple] = [(rt_root, [])]
    while stack:
        node, path = stack.pop()
        if node.replacement is not None:
            reverse_map[node.replacement] = path
        for tok, child in node.children.items():
            stack.append((child, path + [tok]))
    return reverse_map


def decompress(tokens: List[str], reverse_map: Dict[str, List[str]]) -> List[str]:
    """Expand replacement tokens (e.g. "<R0>") back into their original token
    sequence using `reverse_map`, recursively resolving nested replacements."""
    out: List[str] = []
    for tok in tokens:
        if tok in reverse_map:
            out.extend(decompress(reverse_map[tok], reverse_map))
        else:
            out.append(tok)
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


# Bundled pretrained trie, used when no `trie_path` is given.
_DEFAULT_TRIE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bin",
    "trie_pubchem100K",
    "trie_pubchem100K.pkl",
)

EMBEDDING_OFFSET = 10
UNKNOWN_TOKEN = "<unk>"


class TrieTokenizer:
    """
    Longest-match SMILES tokenizer using a pretrained replacement trie.


    Args:
        trie_path: Path to the pickled trie `_State` (as produced by
            `train_trie.py` from https://github.com/BlastCoder/SMILES-Tokenization).
            Defaults to the bundled "bin/trie_pubchem100K/trie_pubchem100K.pkl".
    """

    def __init__(self, trie_path: Optional[str] = None):
        if trie_path is None:
            trie_path = _DEFAULT_TRIE_PATH
        with open(trie_path, "rb") as f:
            state: _State = _RestrictedTrieUnpickler(f).load()
        self.replace_root = state.replace_root
        self._reverse_map: Optional[Dict[str, List[str]]] = None

        self.vocab: List[str] = []
        self.vocab_dict: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}

    def _get_token_index(self, token: str) -> int:
        """Return this token's index, assigning it the next free index on first use."""
        if token not in self.vocab_dict:
            idx = len(self.vocab) + EMBEDDING_OFFSET
            self.vocab.append(token)
            self.vocab_dict[token] = idx
            self.idx_to_token[idx] = token
        return self.vocab_dict[token]

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string, replacing frequent token runs via the trie."""
        return compress(tokenize(smiles), self.replace_root)

    def encode(self, smiles: str) -> List[int]:
        """Tokenize and map to (dynamically assigned) vocabulary indices."""
        return [self._get_token_index(tok) for tok in self.tokenize(smiles)]

    def decode(self, token_ids: List[int]) -> str:
        """Map indices back to tokens, expand replacements and reassemble the SMILES string."""
        tokens = [self.idx_to_token.get(idx, UNKNOWN_TOKEN) for idx in token_ids]
        return "".join(self.detokenize(tokens))

    def detokenize(self, tokens: List[str]) -> List[str]:
        """Expand replacement tokens (e.g. "<R0>") back into the original atom-level
        tokens they were substituted for (inverse of `tokenize`)."""
        if self._reverse_map is None:
            self._reverse_map = build_reverse_map(self.replace_root)
        return decompress(tokens, self._reverse_map)
