# Topic Classification - Chitrabasha Task

## Project Structure
```
project/
├── src/
│   ├── train.py        # training pipeline for all 3 experiments
│   ├── inference.py    # inference script for predictions
│   ├── model.py        # custom MLP architecture
│   └── utils.py        # data loading, cleaning, helper functions
├── experiments/        # logs and notes from experimentation
├── final_models/       # saved model weights and artifacts
├── report.pdf          # detailed project report
├── requirements.txt    # dependencies
└── README.md
```

## Setup Instructions

### Environment Setup
```bash
git clone <your-repo-url>
cd chitrabasha_project
pip install -r requirements.txt
```

### Dependencies Installation
```bash
pip install torch torchvision scikit-learn pandas pyarrow numpy scipy
```

## Training Instructions

1. Place `dataset.parquet` in the project root directory
2. Run training:
```bash
cd src
python train.py
```
This will run all 3 experiments sequentially and save models to `final_models/`

## Inference Instructions
```bash
cd src
python inference.py
```

### Input/Output Schema

**Input:** Raw text string(s)
```python
from inference import load_model, predict
model, vectorizer, le, device = load_model()
predictions = predict(["Your text here"], model, vectorizer, le, device)
```

**Output:** List of predicted topic labels
```python
["finance_and_business"]
```

Available topic labels:
- adult_content, art_and_design, crime_and_law, education_and_jobs
- electronics_and_hardware, entertainment, fashion_and_beauty
- finance_and_business, food_and_dining, games, health
- history_and_geography, home_and_hobbies, industrial, literature
- politics, religion, science_math_and_technology, social_life
- software, software_development, sports_and_fitness
- transportation, travel_and_tourism

## Reproducibility
- Random seed fixed at 42 across numpy, torch, and sklearn
- Same train/val split (80/20, stratified) reproducible with seed=42
- Dataset sampled using fixed batch indices [0, 10, 20, 30]

## Results Summary

| Experiment | Model | Accuracy | Macro F1 |
|---|---|---|---|
| 1 | SGD + HashingVectorizer (CPU baseline) | 88.70% | 0.79 |
| 2 | Custom MLP 33M params (GPU) | 90.78% | 0.85 |
| 3 | Improved MLP 67M params (GPU) | 91.02% | 0.85 |

**Final Model:** Experiment 3 - Improved Custom MLP
- Parameters: 67,775,768 (~67M, well within 5B limit)
- Architecture: 4-layer MLP with BatchNorm, ReLU, Dropout
- Features: HashingVectorizer bigrams (2^16 features)
- Hardware: Tesla T4 GPU
- Training Time: ~227s per epoch x 8 epochs
