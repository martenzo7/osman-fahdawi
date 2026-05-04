"""
Sentiment Analysis Model - Bidirectional LSTM with Attention
Author: Osman
Description: Deep learning model for binary sentiment classification
             trained on movie reviews (positive/negative).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.
    Computes a weighted sum of LSTM hidden states.
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_dim * 2)
        energy = torch.tanh(self.attn(hidden_states))  # (batch, seq_len, hidden_dim)
        attention_scores = self.v(energy).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attention_weights


class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM with Attention for Sentiment Analysis.

    Architecture:
        Embedding → BiLSTM → Attention → Dropout → FC → Sigmoid
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2,
                 dropout=0.5, pad_idx=0):
        super(SentimentLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))        # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)                  # (batch, seq_len, hidden*2)
        context, attn_weights = self.attention(lstm_out)   # (batch, hidden*2)
        out = self.dropout(context)
        logit = self.fc(out).squeeze(-1)                   # (batch,)
        return torch.sigmoid(logit), attn_weights
