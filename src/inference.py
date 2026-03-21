import torch
import pickle
import re
import numpy as np
from model import ImprovedTextClassifier

def clean_text(text, max_len=500):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

def load_model(model_path="final_models/best_model.pt",
               vectorizer_path="final_models/vectorizer_final.pkl",
               encoder_path="final_models/label_encoder.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedTextClassifier(input_dim=65536, num_classes=24).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    le = pickle.load(open(encoder_path, "rb"))
    return model, vectorizer, le, device

def predict(texts, model, vectorizer, le, device):
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = []
    for i in range(len(cleaned)):
        x = torch.FloatTensor(X[i].toarray()).to(device)
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(1).item()
        preds.append(le.classes_[pred])
    return preds

if __name__ == "__main__":
    model, vectorizer, le, device = load_model()
    sample_texts = [
        "The stock market crashed today due to inflation fears.",
        "Scientists discovered a new species of bird in the Amazon.",
        "The football team won the championship after a thrilling match."
    ]
    predictions = predict(sample_texts, model, vectorizer, le, device)
    for text, pred in zip(sample_texts, predictions):
        print(f"Text: {text[:60]}...")
        print(f"Predicted Topic: {pred}\n")
