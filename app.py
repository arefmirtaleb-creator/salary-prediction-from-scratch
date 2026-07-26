import os
import joblib
import streamlit as st
import pandas as pd


# Set page configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered"
)


# Load trained model
def load_model():
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "models", "salary_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            f"Please run 'python src/train.py' first."
        )

    return joblib.load(model_path)


# Main app
def main():
    st.title("Salary Prediction App 💰")
    st.write("Predict salary based on years of experience.")

    try:
        model = load_model()

        years_experience = st.number_input(
            "Enter Years of Experience",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.5
        )

        if st.button("Predict Salary"):
            input_data = pd.DataFrame([[years_experience]], columns=["YearsExperience"])
            prediction = model.predict(input_data)[0]

            st.success(f"Estimated Salary: ${prediction:,.2f}")

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")


if __name__ == "__main__":
    main()
