# Titanic Data Analysis & Visualization Project

A comprehensive data analysis and visualization project using the famous Titanic dataset. This project demonstrates data exploration, preprocessing, visualization, and predictive modeling techniques in Python.

## Project Overview

This project includes:
- Exploratory data analysis of the Titanic passenger data
- Data preprocessing and feature engineering
- Visualization of survival patterns based on different variables
- Building a predictive model to determine factors influencing survival
- Creating a comprehensive dashboard of insights

## Features

- **Data Exploration**: Examine the structure, missing values, and statistical properties of the dataset
- **Data Visualization**: Create insightful visualizations of:
  - Survival distribution
  - Survival rates by gender
  - Survival rates by passenger class
  - Age distribution of survivors vs. non-survivors
  - Fare distribution by survival status
  - Correlation matrix of numeric variables
- **Predictive Modeling**: Build a Random Forest classifier to predict survival and identify important features
- **Dashboard Creation**: Generate a summary dashboard of all visualizations

## Requirements

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/titanic-analysis.git
cd titanic-analysis
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to perform the analysis:
```bash
python titanic_analysis.py
```

The script will:
1. Download the Titanic dataset (or use a local copy if available)
2. Perform exploratory data analysis
3. Create various visualizations
4. Build and evaluate a prediction model
5. Generate a summary dashboard

## Project Structure

```
titanic-analysis/
│
├── titanic_analysis.py  # Main analysis script
├── requirements.txt     # Required packages
├── README.md           # Project documentation
│
├── data/               # Dataset directory
│   └── titanic.csv     # Titanic dataset (downloaded on first run)
│
└── visualizations/     # Generated visualizations
    ├── survival_counts.png
    ├── survival_by_gender.png
    ├── survival_by_class.png
    ├── age_distribution.png
    ├── fare_distribution.png
    ├── correlation_matrix.png
    ├── feature_importance.png
    ├── confusion_matrix.png
    └── dashboard.png   # Summary dashboard
```

## Results

The analysis reveals several key insights about factors affecting survival on the Titanic:
- Gender was a major factor in survival (women had a much higher survival rate)
- Passengers in higher classes (1st class) had better survival rates
- Age distribution shows children had better chances of survival
- The prediction model achieves approximately 80-85% accuracy

The complete set of visualizations can be found in the `visualizations/` directory after running the script.

## Extending the Project

This project can be extended in several ways:
1. Add more advanced feature engineering
2. Try different machine learning algorithms
3. Create an interactive dashboard using Dash or Streamlit
4. Perform more detailed statistical analysis

## License

MIT