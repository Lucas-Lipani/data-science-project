# Transfer AI – Football Transfer Prediction Platform

Transfer AI is an end-to-end forecasting platform that predicts whether a football player will be transferred and, if so, to which club. The project covers all the required steps from the course's practical work (TP), including:

- Data exploration and cleaning ✅
- Model training and comparison ✅
- Containerization (Docker + Docker Compose) ✅
- Frontend interface for user interaction ✅

The platform is designed to simulate a real-world ML pipeline with separation of concerns between model training, inference, and presentation.

---

## 🧠 Features

- Interactive player search and profile viewer
- Transfer probability prediction (classification model)
- Top 5 predicted destination clubs (XGBoost model)
- Full microservice architecture (frontend/backend separation)

---

## 🛠️ Technologies Used

Each component in this stack was chosen to fulfill a specific need in our project's architecture:

- **Python & Flask** – Flask was used to create a lightweight RESTful API backend, allowing the frontend to communicate with machine learning models in a simple and scalable way. Flask’s modularity and simplicity were ideal for quickly connecting the ML model logic to HTTP endpoints.

- **HTML/CSS/JS** – The frontend was designed using standard web technologies to maximize portability. We opted for a responsive layout to make the platform intuitive, with minimal dependencies for ease of deployment. The HTML handles structure, CSS gives it style, and JS is used to interact with the backend in real-time via AJAX (fetch).

- **Pandas & Scikit-learn** – Used extensively for data loading, preprocessing, transformation pipelines, and evaluation. We relied on Pandas for wrangling CSVs and joining datasets, and Scikit-learn’s `Pipeline`, `ColumnTransformer`, and encoders for preparing data for the ML model. This ensures reusability and compatibility during training and prediction.

- **XGBoost** – The final model used to predict the top 5 most likely destination clubs. We selected XGBoost after comparing several options due to its:
  - High performance on structured/tabular data
  - Built-in handling of categorical data and missing values
  - Robust cross-validation and tree boosting optimization
  - Ability to extract feature importance for future explainability

- **Docker & Docker Compose** – To simulate a real-world architecture, we separated the frontend, backend, and model serving logic using Docker. Docker Compose made it easy to orchestrate these services and ensured that the entire application could be reproduced and tested consistently across environments (student machines, teacher demo, etc.).

---

## 📂 Project Structure

```bash
.
├── backend/                  # Flask API with prediction endpoints
│   └── app.py
├── models/                  # Models, encoders, preprocessors
│   ├── transfer_model.py
│   ├── train_club_prediction_model.py
│   └── saved/ (.pkl files)
├── data/                    # CSV files (players, clubs, transfers...)
├── frontend/                # HTML, CSS, JS
│   └── predict-transfer.html
├── Dockerfile               # Backend Dockerfile
├── docker-compose.yml       # Service definitions
└── README.md                # This file
```

---

## 🧪 Machine Learning – Model Training

We implemented and tested models like AR, MA, ARMA but found they were not suitable for our use case. For transfer prediction, we needed non-time-series classifiers. Therefore, we created two separate models:

### 1. Transfer Classification Model (Binary)
This model predicts whether or not a player will be transferred in the next transfer window. It is based on features such as:
- Age
- Contract duration
- Market value trend
- Performance (e.g., minutes played, goals)

This binary classification model outputs a transfer probability (from 0 to 100%). It is used as a first filter before destination prediction.

The classification model is trained using a simple pipeline and `XGBClassifier`, as shown below:

```python
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])

model.fit(X_train, y_train)
```

### 2. Club Destination Model (Multiclass)
We selected **XGBoost** to predict the most likely destination club (among 400+ possible). It was chosen for its:
- High accuracy
- Robust handling of categorical and missing data
- Scalability with multiclass problems
- Feature importance insights

We implemented and tested models like AR, MA, ARMA but found they were not suitable for our use case. For transfer prediction, we needed non-time-series classifiers. 

We selected **XGBoost** due to its:
- High accuracy
- Robust handling of missing and categorical data
- Feature importance interpretability

### 🔧 Training code example: `train_club_prediction_model.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Feature prep pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Final model
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

model.fit(X_train, y_train)
```

Output files:
- `xgb_club_prediction_model.pkl`
- `club_preprocessor.pkl`
- `club_label_encoder.pkl`

These are used in the API to serve predictions live.

---

## 📡 Prediction Logic – `transfer_model.py`

```python
# Load trained model and encoders
model = joblib.load("xgb_club_prediction_model.pkl")
preprocessor = joblib.load("club_preprocessor.pkl")
label_encoder = joblib.load("club_label_encoder.pkl")

# Apply pre-processing and model prediction
X_processed = preprocessor.transform(input_data)
pred_proba = model.predict_proba(X_processed)[0]
predicted_clubs = label_encoder.inverse_transform(np.argsort(pred_proba)[::-1][:5])
```

This generates the top 5 likely destination clubs and maps the `club_id` to their logos.

---

## 🚀 Running the App with Docker

### 1. Build and run everything
```bash
docker-compose up --build
```

- Backend API: http://localhost:5000
- Frontend: http://localhost:8080/predict-transfer.html

---

## 🧪 What We Achieved ✅

| Step                                  | Status      | Notes |
|---------------------------------------|-------------|-------|
| Clean data and handle missing values  | ✅ Complete | CSVs preprocessed manually and programmatically |
| Train models and compare performance  | ✅ Complete | XGBoost outperformed others |
| Build a frontend                      | ✅ Complete | HTML/JS with responsive layout |
| Containerize services                 | ✅ Complete | Backend + Nginx via Docker Compose |
| Bonus model (transfer yes/no)        | ❌ Not implemented | Could be a binary classifier |

---

<!-- ## 📸 Screenshots
*Add screenshots here before your final presentation!*

--- -->

## 📌 Future Work
- Deploy backend with Render or Fly.io
- Add a dashboard for admins
- Add validation dataset and monitoring
- Implement player profile analytics tab

---

## 👨‍💻 Authors
- Lucas Lipani
- Lucas Alves

---

## 📄 License
MIT License
