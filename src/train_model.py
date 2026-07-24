import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_data

def train_model(data_path, model_save_path):
    """
    Loads data, trains a Linear Regression model, evaluates it, and saves the trained model.
    
    Parameters:
    data_path (str): Path to the CSV dataset.
    model_save_path (str): Path where the trained model (.joblib) will be saved.
    
    Returns:
    LinearRegression: The trained model object.
    """
    # 1. Load the cleaned dataset using our data loader
    df = load_data(data_path)
    # 2. Split features (X) and target variable (y)
    # Reshaping X to 2D array as required by scikit-learn
    X = df[['YearsExperience']]
    y = df['Salary']
    # 3. Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
    # 4. Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # 5. Make predictions on test set to evaluate performance
    y_pred = model.predict(X_test)
    # 6. Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n--- Model Training Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Model Formula: Salary = {model.intercept_:.2f} + {model.coef_[0]:.2f} * YearsExperience")
    print("------------------------------\n")
    # 7. Ensure directory for saving model exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 8. Save the trained model to disk for deployment deployment
    joblib.dump(model, model_save_path)
    print(f"[INFO] Model successfully saved to: {model_save_path}")
    
    return model

# Test block for independent execution
if __name__ == "__main__":
    # Resolve paths dynamically to prevent execution errors
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    test_data_path = os.path.join(project_root, "data", "salary_data.csv")
    test_model_path = os.path.join(project_root, "models", "salary_model.joblib")
    
    try:
        print("Starting model training pipeline...")
        trained_model = train_model(test_data_path, test_model_path)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")