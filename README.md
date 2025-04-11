## ⚽ Transfer Prediction Platform

### 📌 Project Overview
This application predicts the probability of a football player being transferred and estimates the most likely destination clubs. It is built with a Python backend using Flask and a dynamic frontend built in HTML/CSS/JS.

---

### 🧠 ML Model Summary
- **Model Used**: XGBoost Classifier
- **Task**: Binary classification (`was_transferred`)
- **Features**:
  - Age, contract remaining time
  - Market value
  - Recent performance (goals, assists, minutes played)
  - Historical transfers (number and average fee)
  - Physical attributes (height, position)
- **Data Handling**:
  - Handled missing values
  - Applied `StandardScaler` and `OneHotEncoder`
  - Used `SMOTE` to balance classes
- **Evaluation**: `classification_report` (accuracy, precision, recall, F1-score)

---

### 🚀 Getting Started

#### Requirements
- Python 3.10+
- pip
- Docker (for containerization)

#### Installation
```bash
pip install -r requirements.txt
```

#### Run Locally
```bash
python backend/app.py
```
Open `frontend/predict-transfer.html` in your browser.

---

### 🖥️ Frontend
Open the file `frontend/predict-transfer.html` directly or serve it via a static file server.

---

### 📁 File Structure
```
├── backend/
│   └── app.py
├── frontend/
│   └── predict-transfer.html
├── models/
│   └── transfer_model.py
├── train_transfer_model.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

### ✅ Next Steps
- Add support for more models
- Show ROC Curve and Confusion Matrix in frontend
- Expand user customization (e.g. model selection)
- Deploy on cloud (Heroku, Render, etc.)