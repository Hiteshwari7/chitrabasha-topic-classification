# Topic Classification Report
## Chitrabasha Task - LINGO Lab, IIT Gandhinagar

---

## 1. Data Processing

### Data Loading Strategy
The dataset contains 10 million rows in a compressed Parquet format (4GB). Loading the full dataset into memory on a single GPU machine (Google Colab T4, ~12GB RAM) is infeasible. We used PyArrow's `iter_batches` API to stream the file in chunks of 100,000 rows and sampled every 20th batch, yielding a representative sample of 400,000 rows (4% of total data) for all experiments.

### Preprocessing Steps
- Lowercasing all text
- Removing URLs using regex
- Collapsing whitespace
- Truncating text to 500 characters to reduce memory footprint while retaining most topical signal

### Tokenization / Feature Engineering
Two approaches were used across experiments:
- **HashingVectorizer:** avoids building vocabulary matrix in RAM, fixed memory footprint, used in Experiments 1-3
- **TF-IDF (word + character ngrams):** captures subword morphology, crucial for multilingual content, used in Experiment 4

### Multilingual Analysis
A significant finding during EDA: **40.76% of the dataset contains non-ASCII (Hindi/multilingual) content.** This varies heavily by class:
- Most multilingual: travel_and_tourism (95.6%), literature (64.8%), food_and_dining (58.3%)
- Least multilingual: home_and_hobbies (9.8%), adult_content (10.2%), transportation (15.0%)

See `experiments/multilingual_analysis.png` for full distribution.

Key insight: multilingual content alone does not hurt performance — travel_and_tourism is 95.6% multilingual yet achieves F1: 0.99. The combination of multilingual content AND low sample size hurts performance (fashion_and_beauty: 57.8% multilingual + 1,446 samples = F1: 0.71).

Character-level ngrams in Experiment 4 explicitly address multilingual content by capturing subword morphology across scripts.

---

## 2. Exploration & Iteration

### Dataset Analysis
- 10 million rows, 24 topic classes
- Highly imbalanced: education_and_jobs (430K) vs games (7.6K) in 2.5M sample
- 40.76% multilingual content (Hindi dominant)
- No null values

### Experiment 1: SGD Classifier + HashingVectorizer (Baseline)
**Motivation:** Establish a fast CPU baseline using classical ML. SGDClassifier with modified_huber loss is known to work well for text classification and trains in seconds.

**Setup:**
- HashingVectorizer: unigrams, 2^16 features
- SGDClassifier: modified_huber loss, 50 iterations
- Training data: 320,000 samples
- Hardware: CPU

**Results:**
- Accuracy: 88.70% | Macro F1: 0.79 | Training time: 22.1s

**Observations:**
- Surprisingly strong baseline for classical approach
- Poor recall on minority classes (art_and_design: 0.27, fashion_and_beauty: 0.37)
- travel_and_tourism and adult_content near perfect

**What didn't work:** TF-IDF with 100K features caused RAM crashes. Switched to HashingVectorizer.

---

### Experiment 2: Custom MLP (GPU)
**Motivation:** Leverage T4 GPU with neural approach. MLP over bag-of-words features built entirely from scratch.

**Setup:**
- HashingVectorizer: bigrams, 2^16 features
- 3-layer MLP: Linear(65536→512)→BN→ReLU→Dropout(0.3)→Linear(512→256)→BN→ReLU→Dropout(0.3)→Linear(256→24)
- Optimizer: Adam lr=1e-3 | Epochs: 5 | Parameters: 33,693,976

**Results:**
- Accuracy: 90.78% | Macro F1: 0.85 | Training time: ~199s/epoch

**Observations:**
- +2.08% over baseline
- Val accuracy plateaued after epoch 1 — overfitting observed
- Converting sparse matrix to dense caused RAM crash — fixed with SparseDataset

---

### Experiment 3: Improved MLP (GPU)
**Motivation:** Address overfitting from Experiment 2 with deeper architecture and better regularization.

**Setup:**
- HashingVectorizer: bigrams, 2^16 features
- 4-layer MLP: Linear(65536→1024)→BN→ReLU→Dropout(0.4)→Linear(1024→512)→BN→ReLU→Dropout(0.4)→Linear(512→256)→BN→ReLU→Dropout(0.4)→Linear(256→24)
- Optimizer: AdamW lr=2e-3, weight_decay=1e-4 | Scheduler: CosineAnnealingLR | Epochs: 8
- Parameters: 67,775,768

**Results:**
- Accuracy: 91.02% | Macro F1: 0.85 | Training time: ~227s/epoch

**Observations:**
- Consistent improvement across all 8 epochs
- Still improving at epoch 8 — more epochs would help
- See `experiments/training_curves_exp3.png`

---

### Experiment 4: FastText-style (Word + Character NGrams + Logistic Regression)
**Motivation:** Task explicitly encourages FastText. FastText's key innovation is subword (character ngram) features which should help with multilingual content.

**Setup:**
- Word TF-IDF: bigrams, 50,000 features
- Character TF-IDF: 3-5 grams, 50,000 features
- Combined: 100,000 features
- Logistic Regression: saga solver, C=5.0, max_iter=500
- Training time: 1649.7s (~27.5 mins)

**Results:**
- Accuracy: 92.05% | Macro F1: 0.86 | Training time: 1649.7s

**Observations:**
- BEST performing model overall — classical ML beat deep learning!
- Character ngrams explicitly capture Hindi subword morphology
- travel_and_tourism F1: 0.99 — character ngrams helped multilingual classes
- art_and_design still weakest (F1: 0.64) — low sample size persists
- Key insight: for multilingual text, character-level features are very powerful

---

## 3. Architecture Details (Final Model - Experiment 4)

**FastText-style Feature Extraction:**

| Component | Type | Features |
|---|---|---|
| Word TF-IDF | Unigrams + Bigrams | 50,000 |
| Char TF-IDF | 3-5 character ngrams | 50,000 |
| Combined | Concatenated sparse matrix | 100,000 |

**Classifier:** Logistic Regression (saga solver)
- C = 5.0 (regularization)
- max_iter = 500
- No pretrained weights — trained entirely from scratch

**Parameter count:** Logistic Regression has 100,000 × 24 = 2,400,024 parameters — well within 5B limit.

**Design decisions:**
- Character ngrams handle multilingual (Hindi) content natively
- Word ngrams capture semantic meaning
- Logistic Regression: simple, interpretable, no risk of overfitting like neural nets

---

## 4. Training Strategy

- **Train/Val Split:** 80/20 stratified (seed=42)
- **Hardware:** Google Colab Tesla T4 GPU (neural experiments), CPU (classical)
- **Random seeds:** Fixed at 42 for numpy, torch, sklearn

| Experiment | Optimizer | Epochs | Time |
|---|---|---|---|
| 1 - SGD | SGD | 50 iter | 22s |
| 2 - MLP | Adam lr=1e-3 | 5 | ~17 min |
| 3 - MLP | AdamW lr=2e-3 | 8 | ~30 min |
| 4 - FastText | saga C=5.0 | 500 iter | ~27 min |

---

## 5. Evaluation Metrics

### All Experiments Summary

| Experiment | Accuracy | Macro P | Macro R | Macro F1 |
|---|---|---|---|---|
| 1 - SGD Baseline | 88.70% | - | - | 0.79 |
| 2 - Custom MLP | 90.78% | 0.88 | 0.82 | 0.85 |
| 3 - Improved MLP | 91.02% | 0.88 | 0.83 | 0.85 |
| 4 - FastText LR | **92.05%** | **0.89** | **0.85** | **0.86** |

See `experiments/accuracy_comparison.png` for visual comparison.

### Final Model Per-class Highlights (Experiment 4)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| travel_and_tourism | 0.99 | 0.98 | 0.99 |
| adult_content | 1.00 | 0.96 | 0.98 |
| finance_and_business | 0.96 | 0.96 | 0.96 |
| art_and_design | 0.82 | 0.52 | 0.64 |
| games | 0.78 | 0.62 | 0.69 |
| industrial | 0.74 | 0.65 | 0.69 |

See `experiments/per_class_f1.png` and `experiments/confusion_matrix_exp3.png`.

---

## 6. Error Analysis

### Where the model fails
- **art_and_design (F1: 0.64):** Only 1,697 samples — severely underrepresented
- **games (F1: 0.69):** Only 958 samples. Overlaps with software and electronics
- **fashion_and_beauty (F1: 0.71):** 57.8% multilingual + low samples
- **industrial (F1: 0.69):** Technical vocabulary overlaps with science_math_and_technology

### Patterns in misclassification
- Minority classes consistently misclassified as majority classes
- software vs software_development confusion (semantically similar)
- science vs education overlap
- Multilingual content + low sample size = worst performance combination

### Potential improvements
1. **Class balancing:** Weighted cross-entropy or oversampling minority classes
2. **More data:** Use full 10M dataset with chunked training
3. **Language detection:** Separate pipelines for Hindi vs English
4. **Ensemble:** Combine FastText-style with MLP for better minority class recall
5. **More epochs:** Neural models still improving at final epoch

---

## 7. Experiment Summary

| Experiment | Model | Accuracy | Macro F1 | Time |
|---|---|---|---|---|
| 1 | SGD + HashingVectorizer | 88.70% | 0.79 | 22s |
| 2 | Custom MLP (33.7M params) | 90.78% | 0.85 | ~17min |
| 3 | Improved MLP (67.8M params) | 91.02% | 0.85 | ~30min |
| 4 | FastText-style LR (Final) | **92.05%** | **0.86** | ~27min |

**Final model selected: Experiment 4** — highest accuracy, best macro F1, handles multilingual content explicitly via character ngrams, and is highly interpretable.
