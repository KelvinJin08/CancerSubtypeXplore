# CancerSubtypeXplore
A Modular Platform for Multi-Omics Cancer Subtype Prediction and Biomarker Consensus Discovery



## 🧩 Overview

**CancerSubtypeXplore** is a modular, user-friendly platform designed for **multi-omics cancer subtype prediction** and **biomarker discovery**.  
It integrates standardized TCGA datasets with both classical and deep learning models in a unified, no-code environment.  
The system is composed of four functional modules:

1. **Dataset Module** – Curated and standardized TCGA multi-omics datasets (mRNA, DNA methylation, miRNA) across 17 cancer types.  
2. **Machine Learning Module** – Automated benchmarking using classical classifiers (SVM, Random Forest, XGBoost, etc.).  
3. **DIY Deep Learning Module** – Interactive interface for designing and training custom neural networks without coding.  
4. **Biomarker Analysis Module** – Extraction and cross-model comparison of top-ranked biomarkers to identify robust pan-cancer signatures.  



## 📊 System Architecture


**Figure:** Overview of the CancerSubtypeXplore framework.  
The platform integrates standardized multi-omics datasets, machine learning benchmarking, customizable deep learning, and cross-model biomarker analysis.



## 📂 Repository Structure

```
CancerSubtypeXplore/
├── data/
│   ├── dataset_for_pretrain/        # Multi-omics datasets for model pretraining (TCGA projects)
│   └── dataset_for_val_ML/          # Datasets for classical ML model benchmarking
│
├── static/                          # Frontend static assets served by FastAPI
│   ├── index.html                   # Web interface for CancerSubtypeXplore
│   ├── script.js                    # Frontend logic (user interaction, requests)
│   └── style.css                    # Styling for the web interface
│
├── __init__.py                      # Package initialization
├── config.py                        # Configuration file (dataset paths, constants)
├── main.py                          # FastAPI application entry point
├── make_ensembl_to_symbol.py        # Utility script for converting Ensembl IDs to gene symbols
├── models_nn.py                     # Neural network architecture definitions (DIY deep learning module)
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```
---

## 🚀 Quick Start

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Launch the FastAPI backend
```bash
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000 to explore the API.

