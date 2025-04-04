import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
sns.set_context("notebook", font_scale=1.2)

# Create directories for output
os.makedirs('visualizations', exist_ok=True)
os.makedirs('data', exist_ok=True)

def download_titanic_data():
    """Download the Titanic dataset if not already present"""
    try:
        # Try to read the data if it already exists
        if os.path.exists('data/titanic.csv'):
            print("Using existing Titanic dataset...")
            df = pd.read_csv('data/titanic.csv')
            print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        
        # If not, try downloading from different public sources
        print("Downloading Titanic dataset...")
        
        # Try Stanford source first
        try:
            url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
            df = pd.read_csv(url)
            print("Dataset downloaded from Stanford source successfully.")
        except:
            # If that fails, try Kaggle source (cached version)
            print("Stanford source failed. Trying alternative source...")
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
            print("Dataset downloaded from alternative source successfully.")
        
        # Save to local file
        df.to_csv('data/titanic.csv', index=False)
        print(f"Dataset saved with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Fallback to a local sample if download fails
        print("Creating sample data instead...")
        
        # Generate sample data that mimics Titanic dataset structure
        np.random.seed(42)
        n_samples = 891
        
        # Create synthetic data
        survived = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        pclass = np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.3, 0.5])
        sex = np.random.choice(['male', 'female'], size=n_samples, p=[0.65, 0.35])
        age = np.random.normal(30, 14, n_samples)
        age = np.clip(age, 0, 80)
        
        siblings = np.random.choice(range(9), size=n_samples, p=[0.7, 0.15, 0.1, 0.02, 0.01, 0.01, 0.005, 0.004, 0.001])
        parents = np.random.choice(range(7), size=n_samples, p=[0.75, 0.15, 0.05, 0.03, 0.01, 0.005, 0.005])
        fare = pclass * 20 + np.random.normal(10, 5, n_samples) * (4 - pclass)
        fare = np.clip(fare, 0, 512)
        
        # Combine into DataFrame
        sample_data = pd.DataFrame({
            'Survived': survived,
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'Siblings/Spouses': siblings,
            'Parents/Children': parents,
            'Fare': fare
        })
        
        # Save to file
        sample_data.to_csv('data/titanic.csv', index=False)
        return sample_data

def explore_data(df):
    """Basic exploration of the dataset"""
    print("\n--- DATASET OVERVIEW ---")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nDescriptive statistics:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """Preprocess the dataset for analysis"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Check and identify non-numeric columns
    print("\n--- DATASET COLUMNS ---")
    for col in data.columns:
        print(f"{col}: {data[col].dtype}")
    
    # Handle different dataset structures (Stanford vs Kaggle format)
    # Keep only numeric and needed categorical columns
    columns_to_keep = []
    
    # First check which columns are available
    if 'Survived' in data.columns:
        columns_to_keep.append('Survived')
    
    if 'Pclass' in data.columns:
        columns_to_keep.append('Pclass')
    
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        columns_to_keep.append('Sex')
    
    if 'Age' in data.columns:
        data['Age'].fillna(data['Age'].median(), inplace=True)
        columns_to_keep.append('Age')
    
    # Handle different naming conventions for family columns
    if 'Siblings/Spouses' in data.columns and 'Parents/Children' in data.columns:
        data['FamilySize'] = data['Siblings/Spouses'] + data['Parents/Children'] + 1
        columns_to_keep.append('FamilySize')
    elif 'SibSp' in data.columns and 'Parch' in data.columns:
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        columns_to_keep.append('FamilySize')
    
    # Add IsAlone feature
    if 'FamilySize' in data.columns:
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        columns_to_keep.append('IsAlone')
    
    # Add Fare if available
    if 'Fare' in data.columns:
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        columns_to_keep.append('Fare')
    
    # Keep only the columns we processed
    print(f"\nKeeping columns for analysis: {columns_to_keep}")
    return data[columns_to_keep]

def visualize_survival_counts(df):
    """Visualize survival counts"""
    plt.figure(figsize=(10, 6))
    
    # Count survivors and non-survivors
    survival_counts = df['Survived'].value_counts()
    
    # Create bar plot
    ax = sns.barplot(x=survival_counts.index, y=survival_counts.values)
    
    # Customize plot
    plt.title('Survival Distribution', fontsize=16)
    plt.xlabel('Survived (1 = Yes, 0 = No)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels on top of bars
    for i, count in enumerate(survival_counts.values):
        ax.text(i, count + 10, str(count), ha='center', fontsize=12)
    
    # Add percentage labels
    total = sum(survival_counts)
    for i, count in enumerate(survival_counts.values):
        percentage = count / total * 100
        ax.text(i, count / 2, f"{percentage:.1f}%", ha='center', fontsize=12, color='white')
    
    plt.tight_layout()
    plt.savefig('visualizations/survival_counts.png')
    plt.close()

def visualize_survival_by_gender(df):
    """Visualize survival by gender"""
    plt.figure(figsize=(12, 6))
    
    # Create crosstab
    survival_by_gender = pd.crosstab(df['Sex'], df['Survived'])
    
    # Plot stacked bar chart
    survival_by_gender.plot(kind='bar', stacked=True, colormap='viridis')
    
    plt.title('Survival by Gender', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(['Did not survive', 'Survived'], title='Outcome')
    
    plt.tight_layout()
    plt.savefig('visualizations/survival_by_gender.png')
    plt.close()

def visualize_survival_by_class(df):
    """Visualize survival by passenger class"""
    plt.figure(figsize=(12, 6))
    
    # Create crosstab
    survival_by_class = pd.crosstab(df['Pclass'], df['Survived'])
    
    # Plot stacked bar chart
    survival_by_class.plot(kind='bar', stacked=True, colormap='viridis')
    
    plt.title('Survival by Passenger Class', fontsize=16)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(['Did not survive', 'Survived'], title='Outcome')
    
    plt.tight_layout()
    plt.savefig('visualizations/survival_by_class.png')
    plt.close()

def visualize_age_distribution(df):
    """Visualize age distribution by survival"""
    plt.figure(figsize=(12, 6))
    
    # Create KDE plot
    sns.kdeplot(data=df, x='Age', hue='Survived', fill=True, common_norm=False)
    
    plt.title('Age Distribution by Survival', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(['Did not survive', 'Survived'], title='Outcome')
    
    plt.tight_layout()
    plt.savefig('visualizations/age_distribution.png')
    plt.close()

def visualize_fare_distribution(df):
    """Visualize fare distribution by survival"""
    plt.figure(figsize=(12, 6))
    
    # Create boxplot
    sns.boxplot(x='Survived', y='Fare', data=df)
    
    plt.title('Fare Distribution by Survival', fontsize=16)
    plt.xlabel('Survived (1 = Yes, 0 = No)', fontsize=12)
    plt.ylabel('Fare (£)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/fare_distribution.png')
    plt.close()

def visualize_correlation_matrix(df):
    """Visualize correlation matrix"""
    # Select numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
    
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()

def build_prediction_model(df):
    """Build a simple prediction model and visualize feature importance"""
    # Make sure we have required columns
    if 'Survived' not in df.columns:
        print("Error: 'Survived' column not found in dataset.")
        return None, 0.0
    
    # Print the dataframe shape and columns for debugging
    print(f"\nProcessed dataframe shape: {df.shape}")
    print(f"Columns for modeling: {df.columns.tolist()}")
    
    # Check if there are any object/string columns left
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"Warning: Column '{col}' contains non-numeric data and will be dropped.")
            df = df.drop(col, axis=1)
    
    # Ensure we have at least one feature column
    if len(df.columns) <= 1:
        print("Error: Not enough valid feature columns for modeling.")
        return None, 0.0
    
    # Prepare data
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    print(f"Features used for prediction: {X.columns.tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n--- TRAINING PREDICTION MODEL ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    
    # Get feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    # Plot feature importance
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()
    
    return clf, accuracy

def create_summary_dashboard():
    """Create a summary dashboard"""
    # Create a 2x3 grid of plots
    plt.figure(figsize=(20, 15))
    
    # Load all visualizations
    survival_counts = plt.imread('visualizations/survival_counts.png')
    survival_by_gender = plt.imread('visualizations/survival_by_gender.png')
    survival_by_class = plt.imread('visualizations/survival_by_class.png')
    age_distribution = plt.imread('visualizations/age_distribution.png')
    fare_distribution = plt.imread('visualizations/fare_distribution.png')
    feature_importance = plt.imread('visualizations/feature_importance.png')
    
    # Plot in a grid
    plt.subplot(2, 3, 1)
    plt.imshow(survival_counts)
    plt.axis('off')
    plt.title('Survival Distribution', fontsize=14)
    
    plt.subplot(2, 3, 2)
    plt.imshow(survival_by_gender)
    plt.axis('off')
    plt.title('Survival by Gender', fontsize=14)
    
    plt.subplot(2, 3, 3)
    plt.imshow(survival_by_class)
    plt.axis('off')
    plt.title('Survival by Class', fontsize=14)
    
    plt.subplot(2, 3, 4)
    plt.imshow(age_distribution)
    plt.axis('off')
    plt.title('Age Distribution', fontsize=14)
    
    plt.subplot(2, 3, 5)
    plt.imshow(fare_distribution)
    plt.axis('off')
    plt.title('Fare Distribution', fontsize=14)
    
    plt.subplot(2, 3, 6)
    plt.imshow(feature_importance)
    plt.axis('off')
    plt.title('Feature Importance', fontsize=14)
    
    plt.suptitle('Titanic Survival Analysis Dashboard', fontsize=24, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('visualizations/dashboard.png', dpi=300)
    plt.close()

def main():
    print("=== TITANIC DATA ANALYSIS & VISUALIZATION ===")
    
    # Download and load data
    df = download_titanic_data()
    
    # Explore data
    df = explore_data(df)
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Visualize data
    print("\n--- CREATING VISUALIZATIONS ---")
    visualize_survival_counts(df)
    visualize_survival_by_gender(df)
    visualize_survival_by_class(df)
    visualize_age_distribution(df)
    visualize_fare_distribution(df)
    visualize_correlation_matrix(processed_df)
    
    # Build model
    model, accuracy = build_prediction_model(processed_df)
    
    # Create dashboard
    print("\n--- CREATING SUMMARY DASHBOARD ---")
    create_summary_dashboard()
    
    print("\nAnalysis complete! All visualizations saved to 'visualizations/' directory.")
    print(f"Final model accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()