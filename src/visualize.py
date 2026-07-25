import os
import matplotlib.pyplot as plt
import joblib

from data_loader import load_data


def visualize_data_and_model(data_path, model_path):
    """
    Load dataset and trained model, then visualize data points
    along with the regression line.
    """

    # 1. Load the dataset
    df = load_data(data_path)

    # 2. Extract feature and target columns
    X = df[["YearsExperience"]]
    y = df["Salary"]

    # 3. Load the trained model from disk
    model = joblib.load(model_path)

    # 4. Generate salary predictions for all experience values
    y_pred = model.predict(X)

    # 5. Create the plot figure
    plt.figure(figsize=(10, 6))

    # 6. Plot actual data points
    plt.scatter(X, y, color="blue", label="Actual Data")

    # 7. Plot regression line
    plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")

    # 8. Add title and axis labels
    plt.title("Salary vs Years of Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")

    # 9. Show legend
    plt.legend()

    # 10. Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.6)

    # 11. Adjust layout
    plt.tight_layout()

    # 12. Display the plot
    plt.show()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(BASE_DIR, "data", "salary_data.csv")
    model_file = os.path.join(BASE_DIR, "models", "salary_model.pkl")

    visualize_data_and_model(data_file, model_file)
