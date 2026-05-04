import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.datasets import fetch_20newsgroups  # placeholder corpus
from sklearn.model_selection import train_test_split
from model.model import SentimentLSTM
from model.vocabulary import Vocabulary, tokenize
from model.dataset import get_dataloaders

CFG = {
    "max_len":      256,
    "batch_size":   64,
    "embed_dim":    128,
    "hidden_dim":   256,
    "num_layers":   2,
    "dropout":      0.5,
    "lr":           1e-3,
    "epochs":       10,
    "patience":     3,          # early-stopping patience
    "max_vocab":    20000,
    "min_freq":     2,
    "save_dir":     "checkpoints",
}
os.makedirs(CFG["save_dir"], exist_ok=True)

def load_synthetic_data(n=2000):
    import random, string
    random.seed(42)
    positives = [
        "this movie was absolutely fantastic and I loved every moment of it",
        "great film with brilliant acting and a wonderful storyline",
        "one of the best movies I have ever seen highly recommend",
        "superb performances and an engaging plot from start to finish",
        "a masterpiece of cinema with stunning visuals and deep emotion",
    ]
    negatives = [
        "terrible film waste of time and money avoid at all costs",
        "boring plot bad acting and completely predictable ending",
        "one of the worst movies I have ever seen do not bother",
        "dull uninteresting and poorly written complete disappointment",
        "awful script with no character development whatsoever",
    ]
    texts, labels = [], []
    for _ in range(n):
        if random.random() > 0.5:
            texts.append(random.choice(positives))
            labels.append(1)
        else:
            texts.append(random.choice(negatives))
            labels.append(0)
    return texts, labels


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            preds, _ = model(seqs)
            loss = criterion(preds, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += ((preds >= 0.5).float() == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    texts, labels = load_synthetic_data(2000)
    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vocab = Vocabulary(CFG["max_vocab"], CFG["min_freq"])
    vocab.build([tokenize(t) for t in tr_texts])
    vocab.save(os.path.join(CFG["save_dir"], "vocab.json"))

    with open(os.path.join(CFG["save_dir"], "config.json"), "w") as f:
        json.dump({**CFG, "vocab_size": len(vocab)}, f, indent=2)

    tr_loader, va_loader = get_dataloaders(
        tr_texts, tr_labels, va_texts, va_labels,
        vocab, CFG["max_len"], CFG["batch_size"]
    )

    model = SentimentLSTM(
        vocab_size=len(vocab),
        embed_dim=CFG["embed_dim"],
        hidden_dim=CFG["hidden_dim"],
        num_layers=CFG["num_layers"],
        dropout=CFG["dropout"],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=CFG["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_acc, patience_count = 0.0, 0

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, tr_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, va_loader, criterion, optimizer, device, train=False)
        scheduler.step(va_loss)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {va_loss:.4f} Acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(CFG["save_dir"], "best_model.pt"))
            print(f"  ✓ Saved best model (val acc {va_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
