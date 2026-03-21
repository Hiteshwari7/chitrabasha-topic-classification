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
We used sklearn's `HashingVectorizer` with the following design decisions:
- **Hashing trick:** avoids building a vocabulary matrix in RAM, fixed memory footprint regardless of data size
- **n_features = 2^16 (65536):** balances expressiveness and memory
- **ngram_range = (1,2):** captures both single words and two-word phrases
- **norm = l2:** ensures consistent feature scaling

---

## 2. Exploration & Iteration

### Dataset Analysis
- 10 million rows, 24 topic classes
- Highly imbalanced: education_and_jobs (430K) vs games (7.6K) in 2.5M sample
- Multilingual content observed (Hindi text present)
- No null values

### Experiment 1: SGD Classifier + HashingVectorizer (Baseline)
**Motivation:** Establish a fast CPU baseline using classical ML. SGDClassifier with modified_huber loss is known to work well for text classification and trains in seconds even on large datasets.

**Setup:**
- HashingVectorizer: unigrams, 2^16 features
- SGDClassifier: modified_huber loss, 50 iterations
- Training data: 320,000 samples
- Hardware: CPU

**Results:**
- Accuracy: 88.70%
- Macro F1: 0.79
- Training time: 22.1 seconds

**Observations:**
- Surprisingly strong baseline for a purely classical approach
- Poor recall on minority classes (art_and_design: 0.27, fashion_and_beauty: 0.37)
- travel_and_tourism and adult_content near perfect (distinctive vocabulary)

**What didn't work:** TF-IDF with 100K features caused RAM crashes on Colab due to dense matrix construction. Switched to HashingVectorizer which uses fixed memory.

---

### Experiment 2: Custom MLP (GPU)
**Motivation:** Leverage the T4 GPU for a neural approach. A simple MLP over bag-of-words features has been shown to be surprisingly competitive for topic classification. Built entirely from scratch with no pretrained weights.

**Setup:**
- HashingVectorizer: bigrams, 2^16 features
- 3-layer MLP: Linear(65536→512)→BN→ReLU→Dropout(0.3)→Linear(512→256)→BN→ReLU→Dropout(0.3)→Linear(256→24)
- Optimizer: Adam, lr=1e-3
- Batch size: 512, Epochs: 5
- Parameters: 33,693,976

**Results:**
- Accuracy: 90.78%
- Macro F1: 0.85
- Training time: ~199s per epoch

**Observations:**
- Significant improvement over baseline (+2.08%)
- Validation accuracy plateaued after epoch 1 (0.9068 → 0.9078) while training accuracy kept rising — classic overfitting
- Key insight: model capacity is sufficient but regularization needs tuning

**What didn't work:** Converting full sparse matrix to dense array caused RAM crash. Fixed by implementing a SparseDataset that converts one row at a time to dense on-the-fly during batching.

---

### Experiment 3: Improved MLP (Final Model)
**Motivation:** Address overfitting observed in Experiment 2 by increasing model capacity, adding stronger regularization, and using a better optimizer and scheduler.

**Changes from Experiment 2:**
- Deeper architecture: 4 layers (added 1024-dim layer)
- Dropout increased: 0.3 → 0.4
- Optimizer: Adam → AdamW (weight decay = 1e-4)
- Scheduler: CosineAnnealingLR (smoother lr decay)
- Epochs: 5 → 8

**Setup:**
- Architecture: Linear(65536→1024)→BN→ReLU→Dropout(0.4)→Linear(1024→512)→BN→ReLU→Dropout(0.4)→Linear(512→256)→BN→ReLU→Dropout(0.4)→Linear(256→24)
- Parameters: 67,775,768

**Results:**
- Accuracy: 91.02%
- Macro F1: 0.85
- Training time: ~227s per epoch
- Best model saved at epoch 8 (still improving — more epochs would help)

**Observations:**
- Consistent improvement across all 8 epochs — model did not overfit as quickly
- CosineAnnealingLR helped maintain stable training
- Still struggling with minority classes

---

## 3. Architecture Details (Final Model)

| Layer | Input Dim | Output Dim | Parameters |
|---|---|---|---|
| Linear 1 | 65536 | 1024 | 67,108,864 |
| BatchNorm 1 | 1024 | 1024 | 2,048 |
| Linear 2 | 1024 | 512 | 524,288 |
| BatchNorm 2 | 512 | 512 | 1,024 |
| Linear 3 | 512 | 256 | 131,072 |
| BatchNorm 3 | 256 | 256 | 512 |
| Linear 4 | 256 | 24 | 6,144 |
| **Total** | | | **67,773,952** |

**Parameter count: 67,775,768 — well within the 5B limit.**

**Design decisions:**
- MLP over sparse BoW features: simple, interpretable, GPU-efficient
- BatchNorm before activation: stabilizes training, reduces internal covariate shift
- Dropout(0.4): prevents overfitting on imbalanced classes
- No pretrained embeddings: built entirely from scratch as required

---

## 4. Training Strategy

- **Train/Val Split:** 80/20 stratified split (seed=42)
- **Batch size:** 512
- **Epochs:** 8
- **Optimizer:** AdamW (lr=2e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR (T_max=8)
- **Hardware:** Google Colab Tesla T4 GPU
- **Training time:** ~227s per epoch (~30 mins total)
- **Random seeds:** Fixed at 42 for numpy, torch, sklearn

---

## 5. Evaluation Metrics

### Final Model (Experiment 3)

| Metric | Score |
|---|---|
| Accuracy | 91.02% |
| Macro Precision | 0.88 |
| Macro Recall | 0.83 |
| Macro F1 | 0.85 |
| Weighted F1 | 0.91 |

### Per-class highlights

| Class | Precision | Recall | F1 |
|---|---|---|---|
| travel_and_tourism | 0.99 | 0.98 | 0.98 |
| adult_content | 1.00 | 0.97 | 0.98 |
| finance_and_business | 0.96 | 0.95 | 0.95 |
| art_and_design | 0.78 | 0.47 | 0.59 |
| games | 0.76 | 0.60 | 0.67 |
| fashion_and_beauty | 0.81 | 0.60 | 0.69 |

---

## 6. Error Analysis

### Where the model fails
- **art_and_design (F1: 0.59):** Only 1,697 samples in 500K — severely underrepresented. Model lacks enough examples to learn distinctive patterns.
- **games (F1: 0.67):** Only 958 samples. Gaming content likely overlaps with software and electronics.
- **fashion_and_beauty (F1: 0.69):** Overlaps with social_life and entertainment vocabulary.
- **industrial (F1: 0.68):** Technical vocabulary overlaps with science_math_and_technology.

### Patterns in misclassification
- Minority classes consistently misclassified as majority classes
- Semantically similar topics confused: software vs software_development, science vs education
- Multilingual content (Hindi) may hurt performance since HashingVectorizer treats all characters equally

### Potential improvements
1. **Class balancing:** Oversample minority classes or use weighted cross-entropy loss
2. **More data:** Use full 10M dataset with chunked training
3. **Character-level features:** Better handle multilingual content
4. **Ensemble:** Combine SGD baseline with MLP for better minority class recall
5. **More epochs:** Experiment 3 was still improving at epoch 8

---

## 7. Experiment Summary

| Experiment | Model | Accuracy | Macro F1 | Params | Time |
|---|---|---|---|---|---|
| 1 | SGD + HashingVectorizer | 88.70% | 0.79 | N/A | 22s |
| 2 | Custom MLP | 90.78% | 0.85 | 33.7M | ~17min |
| 3 | Improved MLP (Final) | **91.02%** | **0.85** | **67.8M** | ~30min |
