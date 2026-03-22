# Experiment Log

## Experiment 1: SGD + HashingVectorizer (Baseline)
- Date: 21-03-2026
- Model: SGDClassifier (modified_huber loss)
- Features: HashingVectorizer unigrams, 2^16 features
- Train size: 320,000 | Val size: 80,000
- Training time: 22.1s
- Val Accuracy: 88.70%
- Macro F1: 0.79
- Notes: RAM crashed with TF-IDF 100K features. Switched to HashingVectorizer.
  Strong baseline. Poor recall on minority classes (art_and_design: 0.27)

## Experiment 2: Custom MLP (GPU)
- Date: 21-03-2026  
- Model: 3-layer MLP from scratch
- Features: HashingVectorizer bigrams, 2^16 features
- Parameters: 33,693,976
- Optimizer: Adam lr=1e-3
- Epochs: 5 | Batch size: 512
- Training time: ~199s/epoch
- Val Accuracy: 90.78%
- Macro F1: 0.85
- Notes: Converting sparse to dense caused RAM crash. Fixed with SparseDataset.
  Val accuracy plateaued after epoch 1 — overfitting observed.
  Significant improvement over baseline (+2.08%)

## Experiment 3: Improved MLP (Final Model)
- Date: 21-03-2026
- Model: 4-layer MLP from scratch (deeper + better regularization)
- Features: HashingVectorizer bigrams, 2^16 features  
- Parameters: 67,775,768
- Optimizer: AdamW lr=2e-3, weight_decay=1e-4
- Scheduler: CosineAnnealingLR T_max=8
- Epochs: 8 | Batch size: 512
- Training time: ~227s/epoch
- Val Accuracy: 91.02%
- Macro F1: 0.85
- Notes: Consistent improvement across all 8 epochs. Still improving at epoch 8.
  CosineAnnealingLR helped maintain stable training.
  Best model saved at epoch 8.

## Summary
| Exp | Model | Accuracy | Macro F1 | Params |
|-----|-------|----------|----------|--------|
| 1 | SGD Baseline | 88.70% | 0.79 | N/A |
| 2 | Custom MLP | 90.78% | 0.85 | 33.7M |
| 3 | Improved MLP | 91.02% | 0.85 | 67.8M |

## Multilingual Analysis
- 40.76% of dataset contains non-ASCII (Hindi/multilingual) content
- travel_and_tourism: 95.6% multilingual → yet best F1 (0.98) due to large sample size
- fashion_and_beauty: 57.8% multilingual + low samples → poor F1 (0.69)
- Key insight: multilingual content + low sample size = poor performance
- HashingVectorizer handles unicode characters natively which explains
  why multilingual classes still perform reasonably well

## Multilingual Analysis
- 40.76% of dataset contains non-ASCII content
- Most multilingual class: travel_and_tourism (95.6%)
- Least multilingual class: home_and_hobbies (9.8%)
- Key insight: multilingual content + low sample size = poor performance
- HashingVectorizer handles unicode natively which helps multilingual classes

## Experiment 4: FastText-style (Word + Character NGrams + Logistic Regression)
- Date: 22-03-2026
- Model: TF-IDF (word bigrams + char 3-5grams) + Logistic Regression
- Word features: 50,000 | Char features: 50,000 | Combined: 100,000
- Optimizer: saga, C=5.0, max_iter=500
- Training time: 1649.7s (~27.5 mins)
- Val Accuracy: 92.05%
- Macro F1: 0.86
- Notes: BEST performing model overall! Classical ML beat deep learning.
  Character ngrams capture subword morphology — crucial for Hindi/multilingual content.
  travel_and_tourism F1: 0.99 (95.6% multilingual — char ngrams helped!)
  art_and_design still weakest (F1: 0.64) — low sample size issue persists.
  Key insight: for multilingual text, character-level features are very powerful.
