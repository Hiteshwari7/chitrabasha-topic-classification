import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
import time
import random

from model import ImprovedTextClassifier
from utils import load_and_prepare_data, save_artifacts

# fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PARQUET_PATH = "dataset.parquet"
SAVE_PATH = "final_models"
BATCH_SIZE = 512
EPOCHS = 8
LR = 2e-3
NUM_CLASSES = 24
INPUT_DIM = 2**16

class SparseDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx].toarray()).squeeze(0)
        return x, self.y[idx]

def experiment1_sgd(X_train, X_val, y_train, y_val, le):
    print("\n=== Experiment 1: SGD + HashingVectorizer ===")
    vectorizer = HashingVectorizer(n_features=2**16, ngram_range=(1,1), alternate_sign=False, norm="l2")
    X_tr = vectorizer.transform(X_train)
    X_v = vectorizer.transform(X_val)
    sgd = SGDClassifier(loss="modified_huber", max_iter=50, random_state=SEED, n_jobs=1)
    start = time.time()
    sgd.fit(X_tr, y_train)
    print(f"Training time: {time.time()-start:.1f}s")
    y_pred = sgd.predict(X_v)
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    pickle.dump(sgd, open(f"{SAVE_PATH}/sgd_model.pkl", "wb"))
    pickle.dump(vectorizer, open(f"{SAVE_PATH}/vectorizer1.pkl", "wb"))
    return sgd, vectorizer

def experiment2_mlp(X_train, X_val, y_train, y_val, le, epochs=5, hidden=512):
    print("\n=== Experiment 2: Custom MLP ===")
    from model import ImprovedTextClassifier
    vectorizer = HashingVectorizer(n_features=INPUT_DIM, ngram_range=(1,2), alternate_sign=False, norm="l2")
    X_tr = vectorizer.transform(X_train)
    X_v = vectorizer.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(INPUT_DIM, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden//2, NUM_CLASSES)
    ).to(device)

    train_loader = DataLoader(SparseDataset(X_tr, np.array(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SparseDataset(X_v, np.array(y_val)), batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = sum((model(X_b.to(device)).argmax(1)==y_b.to(device)).sum().item()
                      for X_b, y_b in val_loader)
        print(f"Epoch {epoch+1}/{epochs} Val Acc: {correct/len(y_val):.4f}")

    torch.save(model.state_dict(), f"{SAVE_PATH}/mlp_model.pt")
    return model, vectorizer

def experiment3_improved_mlp(X_train, X_val, y_train, y_val, le):
    print("\n=== Experiment 3: Improved MLP (Final Model) ===")
    vectorizer = HashingVectorizer(n_features=INPUT_DIM, ngram_range=(1,2), alternate_sign=False, norm="l2")
    X_tr = vectorizer.transform(X_train)
    X_v = vectorizer.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedTextClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(SparseDataset(X_tr, np.array(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SparseDataset(X_v, np.array(y_val)), batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0, 0
        start = time.time()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1)==y_batch).sum().item()
        scheduler.step()
        train_acc = correct/len(y_train)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                out = model(X_batch.to(device))
                val_correct += (out.argmax(1)==y_batch.to(device)).sum().item()
        val_acc = val_correct/len(y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{SAVE_PATH}/best_model.pt")
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | {time.time()-start:.1f}s")

    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    pickle.dump(vectorizer, open(f"{SAVE_PATH}/vectorizer_final.pkl", "wb"))
    return model, vectorizer

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, le = load_and_prepare_data(PARQUET_PATH)
    pickle.dump(le, open(f"{SAVE_PATH}/label_encoder.pkl", "wb"))
    experiment1_sgd(X_train, X_val, y_train, y_val, le)
    experiment2_mlp(X_train, X_val, y_train, y_val, le)
    experiment3_improved_mlp(X_train, X_val, y_train, y_val, le)
