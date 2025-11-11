# 👗 Vibe Matcher — Smart Fashion Recommender

## 🧠 Overview
**Vibe Matcher** is a mini AI-powered recommendation system that matches *fashion products* to a user’s **vibe description** (for example: *“energetic urban chic”* or *“cozy boho weekend”*).  
It uses **Hugging Face Sentence Transformers** to generate semantic embeddings of product descriptions and vibe queries, computes **cosine similarity**, and returns the **Top-3 best-matching items**.

This project demonstrates how semantic embeddings can enable creative, personalized product recommendations using simple vector search.

---

## 🚀 Features

- ✅ **Semantic Matching** with `multi-qa-MiniLM-L6-cos-v1`
- ✅ **Dynamic Similarity Threshold** via Streamlit slider  
- ✅ **Accuracy & Mean Similarity Evaluation**
- ✅ **Normalized Similarity Scores** (0–1 range)
- ✅ **Tag Augmentation** for richer embeddings  
- ✅ **Interactive Streamlit UI**

---

## 🏗️ Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit |
| **Language** | Python 3 |
| **Model** | Hugging Face `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| **Libraries** | `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`, `streamlit` |
| **Similarity Metric** | Cosine Similarity |
| **Evaluation Metric** | Percentage of Top-3 matches above threshold |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/vibe-matcher.git
cd vibe-matcher
streamlit run assignmet.py 
