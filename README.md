#  Fake Review Detection in E-Commerce Applications

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/machine--learning-XGBoost%2C%20RandomForest-success)

##  Overview

This project proposes a **multi-criteria fake review detection system** that leverages **machine learning and contextual analysis** to identify deceptive content on e-commerce platforms. With the rise of fake reviews, the system aims to restore trust and transparency in online product feedback.

The system employs:
- **XGBoost** for binary classification (Fake/Real)
- **Random Forest** for credibility scoring
- **Cosine similarity & contextual scoring** to evaluate review alignment and detect anomalies

An interactive **Flask API backend** and potential browser extension/UI allow real-time usability.

---

##  Features

-  **Fake vs. Real Classification**
-  **Credibility Scoring (Regression)**
-  **Contextual Review Scoring**
-  **Advanced Feature Engineering**
-  **TF-IDF, VADER Sentiment, POS analysis**
-  **Product-Level Reliability Grading (A–E)**
-  **Flask Backend for Real-Time Deployment**
-  **Dashboard / Browser Extension Integration**

---

##  Machine Learning Models

| Task                         | Model             | Accuracy / R² Score |
|------------------------------|-------------------|---------------------|
| Binary Classification        | XGBoost           | **90.96%** Accuracy |
| Credibility Scoring (Regression) | Random Forest     | **R² = 0.99**       |
| Contextual Evaluation        | Rule-based        | Cosine & sentiment analysis |

---

##  Technologies Used

- **Python 3.8+**
- **Scikit-learn**
- **XGBoost**
- **NLTK & TextBlob**
- **VADER Sentiment Analysis**
- **Flask**
- **Pandas, NumPy, Matplotlib**
- **Cosine Similarity (SciPy)**

---

##  System Architecture

```mermaid
flowchart TD
    A[100 Reviews Input] --> B[Preprocessing & Feature Extraction]
    B --> C1[XGBoost Classifier]
    B --> C2[Random Forest Regressor]
    B --> C3[Contextual Scoring Module]
    C1 --> D[Classification Score]
    C2 --> D[Credibility Score]
    C3 --> D[Contextual Score]
    D --> E[Final Score + Grade + Top 5 Reviews]
    E --> F[Flask API JSON Output & CSV]
    F --> G[Dashboard / Browser Extension]


