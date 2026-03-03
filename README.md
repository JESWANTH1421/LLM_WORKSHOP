LLM_WORKSHOP
MiniGPT – A Transformer-Based Language Model Built From Scratch

## 📌 Project Overview

MiniGPT is a lightweight GPT-style Transformer language model built entirely from scratch using PyTorch.

The goal of this project is to understand how Large Language Models (LLMs) work internally by implementing:

- Tokenization
- Self-Attention
- Multi-Head Attention
- Transformer Blocks
- Next-Token Prediction
- Text Generation

This project does NOT use any pretrained models.

---

## 🎯 Objective

To build a simplified GPT-style decoder-only Transformer model capable of:

- Learning language patterns
- Predicting next tokens
- Generating coherent text

---

## 🏗 Architecture

MiniGPT follows the Transformer Decoder architecture:

Input Text  
→ Token Embedding  
→ Positional Encoding  
→ Stacked Transformer Blocks  
    - Masked Multi-Head Self-Attention  
    - Feed Forward Network  
    - Layer Normalization  
→ Linear Output Layer  
→ Softmax  

Training Objective:
Next-token prediction using Cross-Entropy Loss.

---

## 🛠 Tech Stack

- Python 3.x
- PyTorch
- NumPy
- Matplotlib (for visualization)

---

## 📂 Project Structure

MiniGPT/
│
├── data/input.txt
├── model/
│   ├── attention.py
│   ├── transformer.py
│   └── minigpt.py
├── tokenizer.py
├── train.py
├── generate.py
└── README.md

---

## ⚙️ How to Run

### 1️⃣ Install dependencies
pip install torch numpy tqdm matplotlib

### 2️⃣ Train the model
python train.py

### 3️⃣ Generate text
python generate.py

---

## 📊 Training Details

- Model Type: Decoder-only Transformer
- Context Length: Configurable
- Embedding Size: Configurable
- Number of Layers: Configurable
- Loss Function: Cross Entropy
- Optimizer: Adam

---

## 📈 Results

The model successfully learns language structure and generates meaningful text after training.

Sample Output:

Input:
"Machine learning is"

Generated:
"Machine learning is a powerful technique used in data analysis and artificial intelligence..."

---

## 🧠 Concepts Learned

- Transformer Architecture
- Self-Attention Mechanism
- Causal Masking
- Tokenization
- Language Modeling
- Perplexity Evaluation

---

## 🚀 Future Improvements

- Implement Byte Pair Encoding (BPE)
- Add Dropout Regularization
- Train on larger dataset
- Add attention visualization
- Scale model parameters

---

## 🎓 Academic Relevance

This project demonstrates a fundamental understanding of how modern LLMs such as GPT models work internally, without relying on pretrained APIs.

---

## 📌 Author

[jeswanth.k]
Built as part of LLM learning and workshop demonstration.
