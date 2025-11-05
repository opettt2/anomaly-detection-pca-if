# 🔋 Lithium-Ion Battery Anomaly Detection Using PCA & Isolation Forest

This project explores anomaly detection in lithium-ion battery material data using two unsupervised machine-learning techniques:

- **PCA (Principal Component Analysis)** + Mahalanobis Distance  
- **Isolation Forest**

The goal is to detect abnormal or unusual battery behavior patterns to support reliability and early-failure identification — relevant for electric vehicles (EVs), energy storage systems, and portable battery tech.

> 🧪 Developed as part of a college project at Satya Wacana Christian University (UKSW)

---

## 🎯 Objectives

- Clean & preprocess battery materials dataset  
- Apply PCA for dimensionality reduction & anomaly scoring  
- Apply Isolation Forest for anomaly detection  
- Compare anomaly detection results between the two models  

---

## 🧠 Methods

### ✅ Data Preprocessing
- Null handling (mean imputation)
- Feature selection:
  - Formation Energy  
  - Energy Above Hull  
  - Band Gap  
  - Density  
  - Volume  
- Feature scaling using **StandardScaler**

### ✅ PCA + Mahalanobis Distance
- Reduce dataset to 2 principal components  
- Compute Mahalanobis distance to detect outliers  
- Threshold at **95th percentile**

### ✅ Isolation Forest
- Random partitioning to isolate outliers  
- Detect both global & subtle anomalies  

---

## 📊 Results

| Method | Anomalies Detected | Notes |
|---|---|---|
| PCA + Mahalanobis Distance | **4** | Conservative, structured outliers |
| Isolation Forest | **7** | More sensitive, detects subtle patterns |

### Interpretation
- PCA works best with clear distribution separation  
- Isolation Forest is more flexible for complex patterns  

---

## 🧪 Notebook

👉 **Google Colab Notebook:**  
https://colab.research.google.com/drive/1AEyj-duD-D7fiZVyxHlMDVapuftUiN2h?usp=sharing

*(View code, visualizations, and anomaly results there)*

---

## 🛠️ Tech & Tools

- Python  
- NumPy, Pandas, Scikit-Learn  
- Matplotlib / Seaborn  
- Google Colab  

---

## 👤 Author

**Faith Greatfull Samuel Taressy**  
Informatics Engineering — UKSW  
📧 faithtaressy043@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/faithtaressy  
