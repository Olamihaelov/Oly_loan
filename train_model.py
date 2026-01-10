"""
Train Model Script
Trains an SVC model on loan data.
Outputs: model.joblib (trained model), accuracy.txt (accuracy score)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from log import logger

# Model and data paths
MODEL_PATH = "model.joblib"
ACCURACY_PATH = "accuracy.txt"
TRAIN_DATA_PATH = "train.csv"

# Feature columns for model training
FEATURES = ['Married', 'Education', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Log training start
logger.info("="*70)
logger.info("STARTING MODEL TRAINING")
logger.info("="*70)

try:
    # Load training data
    logger.info(f"Loading training data from {TRAIN_DATA_PATH}...")
    df = pd.read_csv(TRAIN_DATA_PATH)
    logger.info(f"Loaded {len(df)} records")

    # Prepare features
    logger.info("Preparing features...")
    X = df[FEATURES].copy()
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # ...existing code...
    X['Married'] = X['Married'].fillna('Yes').map({'Yes': 1, 'No': 0})
    X['Education'] = X['Education'].fillna('Graduate').map({'Graduate': 1, 'Not Graduate': 0})
    X = X.fillna(X.mean())
    logger.info("Features prepared successfully")

    # Split data into train/test sets
    logger.info("Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train set: {len(X_train)} records, Test set: {len(X_test)} records")

    # Train SVC model
    logger.info("Training SVC model with StandardScaler pipeline...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', random_state=42))
    ])
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Calculate accuracy
    logger.info("Calculating accuracy on test set...")
    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Model Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Save model
    logger.info(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved successfully")

    # Save accuracy
    logger.info(f"Saving accuracy to {ACCURACY_PATH}...")
    with open(ACCURACY_PATH, 'w') as f:
        f.write(str(acc))
    logger.info("Accuracy saved successfully")

    # Log completion
    logger.info("="*70)
    logger.info("✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Final Accuracy: {acc:.1%}")
    logger.info(f"Model saved to: {MODEL_PATH}")
    logger.info(f"Accuracy saved to: {ACCURACY_PATH}")
    logger.info("="*70)

    print("\n✅ Training completed! You can now run: streamlit run app2.py")

except Exception as e:
    logger.error(f"❌ Training failed: {str(e)}")
    logger.error("Make sure train.csv exists in the current directory")
    print(f"\n❌ Training failed: {str(e)}")