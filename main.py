import os
import sys
import joblib


def predict_salary(years_experience, model_path):
    """
    Load the trained model and predict salary for a given years of experience.
    """
    # 1. Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'src/train.py' first to train and save the model."
        )

    # 2. Load the trained model
    model = joblib.load(model_path)

    # 3. Format the input for the model
    # Scikit-learn models expect 2D array-like input, e.g., [[value]]
    input_data = [[years_experience]]

    # 4. Make prediction
    predicted_salary = model.predict(input_data)[0]

    return predicted_salary


if __name__ == "__main__":
    # 5. Define dynamic path to the saved model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(BASE_DIR, "models", "salary_model.joblib")

    print("=== Salary Prediction CLI ===")

    # 6. Interactive command line input
    try:
        user_input = input("Enter Years of Experience (e.g., 5.5): ")
        years = float(user_input)

        if years < 0:
            print("[Error] Experience cannot be negative!")
            sys.exit(1)

        # 7. Run prediction
        salary = predict_salary(years, model_file)

        # 8. Print the result nicely formatted
        print("\n--- Prediction Result ---")
        print(f"Experience: {years} years")
        print(f"Predicted Salary: ${salary:,.2f}")
        print("-------------------------\n")

    except ValueError:
        print("[Error] Please enter a valid numerical value for experience.")
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
