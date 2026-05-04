"""
Vocabulary and Text Preprocessing Utilities
"""

import re
import json
from collections import Counter


class Vocabulary:
    """
    Builds and manages a word-level vocabulary from a corpus.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab_size=20000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()

    def build(self, texts):
        """Build vocabulary from a list of tokenized texts."""
        for tokens in texts:
            self.word_counts.update(tokens)

        # Filter by min frequency and size
        vocab = [w for w, c in self.word_counts.most_common(self.max_vocab_size)
                 if c >= self.min_freq]

        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for word in vocab:
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"Vocabulary built: {len(self.word2idx)} tokens")

    def encode(self, tokens):
        unk = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(t, unk) for t in tokens]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)
        print(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "r") as f:
            obj.word2idx = json.load(f)
        obj.idx2word = {idx: word for word, idx in obj.word2idx.items()}
        return obj


def clean_text(text):
    """Basic text normalization."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # keep letters, digits, apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Simple whitespace tokenizer after cleaning."""
    return clean_text(text).split()


def pad_sequence(seq, max_len, pad_idx=0):
    """Pad or truncate a sequence to max_len."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))
