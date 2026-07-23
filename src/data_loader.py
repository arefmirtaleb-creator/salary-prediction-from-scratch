import pandas as pd
import os

def load_data(file_path):
    """
    Loads the salary dataset from a CSV file and validates its structure.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: Cleaned and validated DataFrame containing the data.
    """
    # Check if the file exists at the specified path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The data file was not found at: {file_path}")
    
    # Load the dataset using pandas
    data = pd.read_csv(file_path)
    
    # Define required columns for the salary prediction task
    required_columns = {'YearsExperience', 'Salary'}
    
    # Check if the dataset contains all required columns
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing required columns in dataset: {missing}")
    
    # Drop rows with missing (NaN) values to ensure clean data for training
    clean_data = data.dropna(subset=['YearsExperience', 'Salary'])
    
    # Show a brief summary of loaded data in console for debugging
    print(f"[INFO] Data loaded successfully from {file_path}")
    print(f"[INFO] Dataset shape: {clean_data.shape}")
    
    return clean_data

if __name__ == "__main__":
    # Get the directory of the current script (src folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root and then into the data folder
    project_root = os.path.dirname(current_dir)
    test_path = os.path.join(project_root, "data", "salary_data.csv")
    
    try:
        print(f"Target path: {test_path}") # This will show exactly where it's looking
        print("Testing data loader...")
        df = load_data(test_path)
        print("\nFirst 5 rows of the loaded dataset:")
        print(df.head())
    except Exception as e:
        print(f"\n[ERROR] An error occurred during test: {e}")
