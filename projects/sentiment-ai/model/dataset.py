"""
Dataset and DataLoader for Sentiment Analysis
"""

import torch
from torch.utils.data import Dataset, DataLoader
from model.vocabulary import tokenize, pad_sequence


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment data.

    Args:
        texts  (list[str]): Raw review texts.
        labels (list[int]): 1 = positive, 0 = negative.
        vocab  (Vocabulary): Fitted vocabulary object.
        max_len (int)      : Sequence length (tokens).
    """

    def __init__(self, texts, labels, vocab, max_len=256):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []

        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            indices = vocab.encode(tokens)
            padded = pad_sequence(indices, max_len)
            self.samples.append((padded, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


def get_dataloaders(train_texts, train_labels,
                    val_texts, val_labels,
                    vocab, max_len=256, batch_size=64, num_workers=0):
    """Returns (train_loader, val_loader) DataLoader pair."""
    train_ds = SentimentDataset(train_texts, train_labels, vocab, max_len)
    val_ds   = SentimentDataset(val_texts,   val_labels,   vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
