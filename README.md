# CancerSubtypeXplore
A Modular Platform for Multi-Omics Cancer Subtype Prediction and Biomarker Consensus Discovery



## 🧩 Overview

**CancerSubtypeXplore** is a modular, user-friendly platform designed for **multi-omics cancer subtype prediction** and **biomarker discovery**.  
It integrates standardized TCGA datasets with both classical and deep learning models in a unified, no-code environment.  
The system is composed of four functional modules:

1. **Dataset Module** – Curated and standardized TCGA multi-omics datasets (mRNA, DNA methylation, miRNA) across 17 cancer types.  
2. **Machine Learning Module** – Automated benchmarking using classical classifiers (SVM, Random Forest, etc.).  
3. **Disign Your Deep Learning Module** – Interactive interface for designing and training custom neural networks without coding.  
4. **Biomarker Analysis Module** – Extraction and cross-model comparison of top-ranked biomarkers to identify robust pan-cancer signatures.  



## 🪢  Workflow

<img width="832" height="544" alt="image" src="https://github.com/user-attachments/assets/d70bfd75-d95c-41cb-a29c-f87daac15298" />


**Figure:** Overview of the CancerSubtypeXplore workflow
**a. Dataset preprocessing** – Curated TCGA multi-omics datasets (mRNA, DNA-methylation, miRNA) are presented after prior download, filtering, and normalization, ensuring consistent feature dimensions across projects. **b. Machine-learning baseline** – Classical algorithms (e.g., Logistic Regression, SVM, Random Forest) are benchmarked to establish baseline performance across cancers. **c. Deep-learning model design** – Users can interactively define or customize neural architectures without coding to explore model configurations beyond ML baselines. **d. Cross-model and cross-cancer biomarker discovery** – Contribution-based feature-ranking and frequency aggregation identify stable biomarkers reproducibly highlighted by multiple deep-learning models.



## 📂 Repository Structure

```
CancerSubtypeXplore/
├── data/
│   ├── independent/        # 10 independent TCGA datasets
│   └── cross_cancer/          # 7 cross-cancer TCGA datasets
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
pip install torch --index-url https://download.pytorch.org/whl/cpu             # You can also download the GPU version that matches your computer
```

### 2️⃣ Launch the FastAPI backend
```bash
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000 to explore the API.

## 🖥️ Web Tool Preview

<img width="832" height="990" alt="image" src="https://github.com/user-attachments/assets/e1cd16df-8127-44d0-baab-7d835a45c599" />
<img width="832" height="1010" alt="image" src="https://github.com/user-attachments/assets/f9b2572b-c698-45f5-a526-becc697ad201" />




