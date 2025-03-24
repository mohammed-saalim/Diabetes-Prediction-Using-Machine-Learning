# Diabetes Prediction Using Machine Learning

This project uses machine learning models to predict whether a person is likely to have diabetes based on health-related attributes. The final model is deployed as a web app using Streamlit.

## Dataset
- **Source**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Attributes**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (Target variable)

## Models Used
- Support Vector Machine (SVM)
- Decision Tree (with GridSearchCV)
- Random Forest
- Gradient Boosting
- **Voting Classifier (Ensemble)** – Used for deployment

## Evaluation
All models were evaluated using accuracy, precision, recall, F1-score, and ROC curves. The VotingClassifier showed the best balance and was selected for deployment.

## Deployment
The project is deployed as a **Streamlit web app**, where users can enter their health information and get an instant prediction.

- **Live App**: [Click here to use](https://diabetes-prediction-using-machine-learning-2vjsw3v4gspej6gk2lw.streamlit.app)
- **Try locally**:
  ```bash
  git clone https://github.com/mohammed-saalim/Diabetes-Prediction-Using-Machine-Learning.git
  cd Diabetes-Prediction-Using-Machine-Learning
  pip install -r requirements.txt
  streamlit run app.py

## Features of the App
Real-time prediction of diabetes risk
Probability display with progress bar
Sample inputs (Diabetic / Non-Diabetic)
Reset button
Uses the same scaler (StandardScaler) from training for input preprocessing

## Files
diabetes.csv – Dataset
project.ipynb – Full analysis, training, evaluation
app.py – Streamlit app code
voting_model.pkl – Trained Voting Classifier model
scaler.pkl – StandardScaler object used for prediction

  
