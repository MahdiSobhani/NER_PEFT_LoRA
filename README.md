# NER_PEFT_LoRA
# Persian NER with ParsBERT, Full Fine-tuning & LoRA

A practical Persian Named Entity Recognition (NER) project using **ParsBERT**, with a comparison between **Full Fine-tuning** and **LoRA (Low-Rank Adaptation)**.

## 📌 Project Overview

In this project, a Persian NER dataset is preprocessed and prepared for token classification. A pretrained Persian BERT model, **HooshvareLab/bert-base-parsbert-uncased**, is then adapted to the NER task using two different approaches:

1. **Full Fine-tuning**
2. **LoRA Fine-tuning**

Finally, the models are evaluated and their predictions are compared.

```text
Persian NER Dataset
        ↓
Preprocessing & Token-Label Alignment
        ↓
      ParsBERT
        │
        ├── Full Fine-tuning
        │
        └── LoRA Fine-tuning
                 ↓
        Evaluation & Comparison
```

## 🎯 Objectives

* Understand the preprocessing pipeline for token classification.
* Learn how to align word-level NER labels with BERT subword tokens.
* Fine-tune ParsBERT for Persian NER.
* Apply LoRA using PEFT.
* Compare Full Fine-tuning and LoRA in terms of:

  * Precision
  * Recall
  * F1
  * Accuracy
  * Trainable parameters
  * Prediction agreement

## 📚 Dataset

The project uses:

**HaniehPoostchi/persian_ner**

Dataset structure:

```text
Train: 5121 samples
Test:  2560 samples
```

The dataset contains tokenized Persian sentences and BIO-style NER labels.

### Entity Labels

The project contains 13 labels:

```text
O
B-event    I-event
B-fac      I-fac
B-loc      I-loc
B-org      I-org
B-pers     I-pers
B-pro      I-pro
```

## 🧠 Base Model

The base model is:

```text
HooshvareLab/bert-base-parsbert-uncased
```

ParsBERT is used as the pretrained language model.

The pretrained model already contains learned representations of Persian language and contextual relationships, but it is not specifically fine-tuned for the NER labels used in this project.

A token classification head is added for the 13 NER classes.

## 🔧 Preprocessing

The dataset contains word-level tokens and NER labels.

Because BERT uses subword tokenization, the original word-level labels must be aligned with the generated subword tokens.

The project uses:

```python
tokenizer(
    example["tokens"],
    truncation=True,
    is_split_into_words=True
)
```

`word_ids()` is then used to map each generated token back to its original word.

Special tokens are assigned:

```text
-100
```

so they are ignored during loss calculation.

## 🔥 Full Fine-tuning

In the first experiment, all model parameters are trainable.

```text
ParsBERT
   ↓
All parameters trainable
   ↓
NER Fine-tuning
```

Main configuration:

```text
Learning Rate:       2e-5
Batch Size:          16
Epochs:              3
Weight Decay:        0.01
Mixed Precision:     FP16
Best Model:          F1
```

The best checkpoint is selected based on validation F1.

## ⚡ LoRA Fine-tuning

The second approach uses **LoRA (Low-Rank Adaptation)** through the PEFT library.

Instead of updating all BERT parameters, LoRA adds trainable low-rank matrices to selected layers while keeping most of the pretrained model frozen.

The project also keeps the NER classifier trainable:

```python
modules_to_save=["classifier"]
```

### LoRA Configuration

One experiment uses:

```text
r = 32
alpha = 64
dropout = 0.1
target_modules = query, key, value
```

A second/best LoRA configuration uses:

```text
r = 16
alpha = 32
dropout = 0.1
target_modules =
    query
    key
    value
    intermediate.dense
    output.dense
```

This configuration was trained for 5 epochs with:

```text
Learning Rate: 5e-5
Batch Size:    16
Epochs:        5
```

## 📊 Evaluation

The models are evaluated using:

* Precision
* Recall
* F1 Score
* Accuracy

The F1 score is used as the main metric for selecting the best checkpoint.

Example Full Fine-tuning result:

```text
Epoch 1 → F1: 0.7921
Epoch 2 → F1: 0.8032
Epoch 3 → F1: 0.8057
```

The best Full Fine-tuning model achieved approximately:

```text
F1: 0.806
Accuracy: 0.976
```

## 🆚 Full Fine-tuning vs LoRA

The project compares the two approaches from multiple perspectives.

### 1. Trainable Parameters

Full Fine-tuning:

```text
All model parameters
```

LoRA:

```text
Only LoRA parameters + classifier
```

The LoRA configuration reduces the number of trainable parameters significantly.

### 2. Prediction Comparison

The project evaluates both models on Persian sentences containing:

* Persons
* Locations
* Organizations
* Events
* Products
* Facilities

Example:

```text
علی رضایی مدیر شرکت مایکروسافت در تهران با مسئولان
وزارت ارتباطات ایران دیدار کرد.
```

Both models generate NER predictions, which are then converted into readable entities.

### 3. Entity Agreement

The project also measures how often Full Fine-tuning and LoRA produce the same prediction for actual entities.

The comparison intentionally excludes the `O` label so that the agreement focuses on entity tokens rather than the large number of non-entity tokens.

Example result:

```text
Entity Agreement: 91.15%
```

This indicates a high level of agreement between the predictions of the two approaches on entity tokens.

## 🔍 Prediction Pipeline

The prediction pipeline is:

```text
Input Text
    ↓
Tokenizer
    ↓
ParsBERT
    ↓
Token Classification Head
    ↓
Predicted BIO Labels
    ↓
Entity Merging
    ↓
Readable Entities
```

For example:

```text
Input:
محمد خاتمی در تهران با مسئولان سازمان ملل دیدار کرد
```

The model produces token-level BIO labels, which are then merged into entities such as:

```text
محمد خاتمی → شخص
تهران      → مکان
سازمان ملل → سازمان
```

## 🗂️ Saved Models

The trained models are saved separately.

```text
Ner-full-fine-tune/
Ner-Lora-v2/
Ner-Lora-best/
```

The Full Fine-tuned model is saved as a complete Hugging Face model.

The LoRA model is saved as a PEFT adapter and can be loaded on top of the original base model.

## 🛠️ Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* LoRA
* Evaluate
* Seqeval
* NumPy
* Google Colab

## 📁 Main Workflow

```text
Load Dataset
     ↓
Inspect Labels
     ↓
Load Tokenizer
     ↓
Tokenization & Label Alignment
     ↓
Data Collator
     ↓
Metrics
     ↓
Load ParsBERT
     ↓
┌───────────────────────┐
│ Full Fine-tuning      │
└───────────────────────┘
     ↓
Evaluation
     ↓
Save Model
     
     ┌───────────────────┐
     │ LoRA / PEFT        │
     └───────────────────┘
             ↓
        Evaluation
             ↓
         Save Adapter
             ↓
     Prediction Comparison
             ↓
      Entity Agreement
```

## 🚀 Key Takeaways

This project demonstrates that a pretrained language model can be adapted to a specific NLP task using different fine-tuning strategies.

**Full Fine-tuning** updates the entire model, while **LoRA** updates only a small number of additional parameters.

The project therefore provides a practical comparison between:

```text
Full Fine-tuning
        VS
LoRA / PEFT
```

with respect to model performance, trainable parameters, and prediction behavior.

## 👤 Project Focus

This project was implemented as a practical study of:

```text
Persian NLP
    +
Transformer Fine-tuning
    +
Parameter-Efficient Fine-tuning
    +
LoRA
    +
NER
```
