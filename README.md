# salary-prediction-from-scratch
Salary prediction using simple linear regression in Python, with scikit-learn and a planned from-scratch implementation.
# Salary Prediction App

A simple and interactive **Machine Learning web application** that predicts salary based on **Years of Experience** using **Linear Regression**.  
This project was built as part of my machine learning practice and portfolio development, with a focus on clean project structure, model training, visualization, and deployment using **Streamlit Community Cloud**.

## Live Demo

https://salary-prediction-from-scratch-ncvhlzc6h3y5prfzmxt7qv.streamlit.app/

---

## Project Overview

This project uses a **Linear Regression** model to estimate salary from years of experience.  
It includes:

- Data loading and preprocessing
- Model training with Scikit-learn
- Model saving using Joblib
- Data visualization with regression line
- A simple CLI-based prediction interface
- A web interface built with Streamlit

The goal of this project is to demonstrate a complete beginner-friendly machine learning workflow, from training a model to deploying it online.

---

## Features

- Predict salary based on user input
- Train and save a machine learning model
- Visualize data and regression line
- Run predictions from the command line
- Use a simple web interface with Streamlit
- Ready for cloud deployment

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Joblib**
- **Streamlit**

---

## Project Structure
```bash
salary-prediction/
│
├── app.py                  # Streamlit web app
├── main.py                 # CLI prediction interface
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
│
├── data/
│   └── Salary_Data.csv     # Dataset
│
├── models/
│   └── salary_model.joblib # Trained model
│
└── src/
├── data_loader.py      # Load dataset
├── train.py            # Train and save the model
└── visualize.py        # Visualize data and regression line
