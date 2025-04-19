# ⚙️ Predictive Maintenance for CNC Milling Machines

> 🚀 Boosting uptime and reducing maintenance costs using machine learning for early fault detection in CNC milling operations.

---

## 👩‍💻 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/sarohaanamika">
        <img src="https://github.com/sarohaanamika.png" width="100px;" alt="Anamika Saroha"/><br />
        <sub><b>Anamika Saroha</b></sub><br />
        <sup>Lead Developer</sup>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Aashish-kamath">
        <img src="https://github.com/Aashish-kamath.png" width="100px;" alt="Aashish S Kamath"/><br />
        <sub><b>Aashish S Kamath</b></sub><br />
        <sup>ML Engineer</sup>
      </a>
    </td>
  </tr>
</table>


---


![maintenance](https://img.shields.io/badge/Machine%20Learning-Predictive%20Maintenance-blue?style=for-the-badge&logo=python)
![status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=github)
## FEATURE IMPORTANCE WITH RANDOM FOREST
![PHOTO-2025-04-19-21-53-36](https://github.com/user-attachments/assets/bf053f5c-4399-4816-8ad0-fdc8ed8f3746)


> 🚀 Boosting uptime and reducing maintenance costs using machine learning for early fault detection in CNC milling operations.

---
![PHOTO-2025-04-18-21-12-09](https://github.com/user-attachments/assets/b313d66e-c968-4745-bd1a-cba661ca25eb)

## 📌 Overview

Predictive maintenance is a proactive approach that leverages data-driven models to forecast equipment failures **before** they happen. In this project, we apply machine learning algorithms to monitor CNC milling machine data and accurately predict component failures.

This solution enables:
- 🛡️ Reduced unexpected downtimes
- ⚙️ Optimized maintenance schedules
- 💸 Cost savings in industrial operations

---

---

## 🔍 Dataset Description

The dataset consists of real-time readings from CNC milling machines, including:

| Feature              | Description                          |
|---------------------|--------------------------------------|
| Air temperature (°C) | Ambient air temp around the machine |
| Process temperature  | Temperature during milling          |
| Rotational speed     | RPM of the milling shaft            |
| Torque               | Rotational force applied            |
| Tool wear (min)      | Wear and tear of the tool           |
| Target label         | Binary: Failure (1) or Normal (0)   |

---

## 🧠 ML Techniques Used

- 📊 **Exploratory Data Analysis** (EDA)
- 🧼 **Feature Engineering** (Statistical + Frequency domain)
- 🔍 **Modeling**:
  - Logistic Regression
  - Random Forest
  - XGBoost
- 🧪 **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC

- 🧠 **Model Explainability**:
  - LIME (Local Interpretable Model-agnostic Explanations)
  - SHAP (SHapley Additive exPlanations)

---

## 📈 Performance Snapshot

| Model            | Accuracy | Precision | Recall | AUC   |
|------------------|----------|-----------|--------|-------|
| LogisticRegression | 92.4%   | 0.91      | 0.90   | 0.93  |
| RandomForest     | 95.8%   | 0.96      | 0.94   | 0.97  |
| XGBoost          | **97.2%** | **0.97**   | **0.96** | **0.98** |

✔️ **XGBoost emerged as the top performer**, balancing accuracy with model interpretability.

---

## 📊 Dashboard

An interactive Streamlit dashboard was built to:
- Predict failures in real-time
- Visualize sensor trends
- Interpret predictions using SHAP & LIME

📍 Coming Soon: Live demo deployment on HuggingFace Spaces!

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/sarohaanamika/Predictive_Maintenance_Milling_Machine.git
cd Predictive_Maintenance_Milling_Machine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Jupyter Notebooks
jupyter notebook

# 4. Launch Dashboard (optional)
streamlit run dashboards/app.py
