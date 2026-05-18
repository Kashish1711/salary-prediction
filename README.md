# 💼 Salary Prediction using Simple Linear Regression

A beginner-friendly Machine Learning project that predicts an employee's salary based on their years of experience using Simple Linear Regression.

---

# Project Overview

This project demonstrates how to build a supervised machine learning model using real-world salary data. It covers the complete ML workflow — from data loading and preprocessing to model training, evaluation, and visualization.

---

# Project Structure

```
salary-prediction/
│
├── Salary_dataset.csv      # Dataset with YearsExperience and Salary columns
├── salarydata.py           # Main Python script (model training + evaluation)
├── README.md               # Project documentation
└── .gitignore              # Files to exclude from Git
```

---

# Dataset

| Column            | Description                        |
|-------------------|------------------------------------|
| `Sr. No.`         | Serial number (index)              |
| `YearsExperience` | Years of work experience (float)   |
| `Salary`          | Annual salary in USD (integer)     |

- *Total Records:* 30 entries
- *Experience Range:* 1.2 to 10.6 years
- *Salary Range:* ~$37,000 to ~$1,22,000

---

# Algorithm Used

*Simple Linear Regression*

The model learns a straight-line relationship between experience and salary:

```
Salary = m × YearsExperience + c
```

Where:
- `m` = slope (how much salary increases per year of experience)
- `c` = intercept (base salary with 0 experience)

---

# Tech Stack

| Tool / Library   | Purpose                              |
|------------------|--------------------------------------|
| Python 3.x       | Programming language                 |
| pandas           | Data loading and manipulation        |
| numpy            | Numerical operations                 |
| matplotlib       | Data visualization / plotting        |
| scikit-learn     | ML model training and evaluation     |

---

# How to Run

# 1. Clone the repository
```bash
git clone https://github.com/Kashish1711/salary-prediction.git
cd salary-prediction
```

# 2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
```

# 3. Run the script
```bash
python salarydata.py
```

---

# Model Evaluation Metrics

| Metric | Description                                      |
|--------|--------------------------------------------------|
| MAE    | Mean Absolute Error — average prediction error  |
| MSE    | Mean Squared Error — penalizes large errors     |
| RMSE   | Root MSE — error in salary units (USD)          |
| R²     | Coefficient of determination (closer to 1 = better fit) |

---

# Sample Output (Visualizations)

The script generates two scatter plots with regression lines:
- *Training Set Plot* — how well the model fit the training data
- *Test Set Plot* — how well the model predicts on unseen data

---

# Results

The model achieves a strong R² score (typically ~0.95+), indicating that *years of experience is a strong predictor of salary* for this dataset.

---

# Author

Kashish Gupta
- GitHub: [@Kashish1711](https://github.com/Kashish1711)

---

# License

This project is open source and available under the [MIT License](LICENSE).