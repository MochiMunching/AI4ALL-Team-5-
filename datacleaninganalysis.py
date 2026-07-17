"""
Data Cleaning and Exploratory Data Analysis
AI4ALL Agentic AI Security Risk Dataset

This script:
1. Downloads the dataset
2. Loads the data
3. Examines dataset structure
4. Checks data quality
5. Performs exploratory data analysis
6. Examines possible bias
7. Visualizes important distributions

Load Dataset
│
├── Dataset Overview
│
├── Data Cleaning
│   ├── Duplicate detection
│   ├── Missing values
│   ├── Outlier detection
│
├── Exploratory Data Analysis
│   ├── Summary statistics
│   ├── Categorical distributions
│   ├── Group representation
│   ├── Average risk by agent
│   ├── Risk score histogram
│   └── Access decision bar chart
│
└── Machine Learning
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

from dotenv import load_dotenv
import os

import kagglehub
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use("ggplot")

# ============================================================
# LOAD DATASET
# ============================================================

# Load environment variables (.env file)
load_dotenv()

# Download the latest version of the dataset from KaggleHub
path = kagglehub.dataset_download(
    "algozee/agentic-ai-security-risk-dataset"
)

print("Dataset location:", path)
print("Files:", os.listdir(path))

# Read the CSV file into a pandas DataFrame
csv_file = os.path.join(path, "agent_security_risk_scores.csv")
df = pd.read_csv(csv_file)

# ============================================================
# DATASET OVERVIEW
# ============================================================

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def summary_statistics(df):
    """
    Display summary statistics for both numerical and
    categorical columns.
    """
    print(df.describe(include="all"))


def find_duplicates(df):
    """
    Count duplicate rows in the dataset.
    """
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")


def check_missing_values(df):
    """
    Count missing values in every column.
    """
    missing = df.isnull().sum()

    missing_percent = (
        missing / len(df) * 100
    ).round(2)

    results = pd.DataFrame({
        "Missing Values": missing,
        "Percent": missing_percent
    })

    print(results)


def detect_outliers(df, column):
    """
    Detect outliers using the IQR method.

    Returns rows whose values fall outside
    1.5 × IQR.
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    print(f"Outliers in '{column}': {len(outliers)}")

    return outliers


# Run data quality checks

print("\n========== SUMMARY STATISTICS ==========")
summary_statistics(df)

print("\n========== DUPLICATE ROWS ==========")
find_duplicates(df)

print("\n========== MISSING VALUES ==========")
check_missing_values(df)

print("\n========== OUTLIERS ==========")
outliers = detect_outliers(df, "action_risk_score")
print(outliers)

# ============================================================
# CATEGORICAL FEATURE ANALYSIS
# ============================================================

# Display the frequency of every category
# in each categorical feature.

print("\n========== CATEGORICAL DISTRIBUTIONS ==========")

for column in df.select_dtypes(include="object"):

    # print(f"\n{column}")
    print(df[column].value_counts())

# ============================================================
# FAIRNESS / BIAS ANALYSIS
# ============================================================

def check_group_representation(df, group_column):
    """
    Compute the proportion of each group
    within a categorical feature.
    """
    return df[group_column].value_counts(normalize=True)


def compare_scores_by_agent_role(df):
    """
    Compute the average risk score
    for every agent role.
    """
    return (
        df.groupby("agent_role")["action_risk_score"]
        .mean()
        .sort_values(ascending=False)
    )


print("\n========== AGENT ROLE DISTRIBUTION ==========")
print(check_group_representation(df, "agent_role"))

print("\n========== AVERAGE RISK SCORE BY AGENT ==========")
print(compare_scores_by_agent_role(df))

# ============================================================
# DATA VISUALIZATION
# ============================================================

# Plot the distribution of action risk scores.

plt.figure(figsize=(8,5))

plt.hist(
    df["action_risk_score"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Action Risk Scores")
plt.xlabel("Action Risk Score")
plt.ylabel("Count")

plt.show()

# Plot the distribution of access decisions.

plt.figure(figsize=(7,5))

df["access_decision"].value_counts().plot(
    kind="bar"
)

plt.title("Access Decision Distribution")
plt.xlabel("Access Decision")
plt.ylabel("Count")

plt.show()

#Plot the distribution of agent roles.

plt.figure(figsize=(10,5))

(
    df["agent_role"]
    .value_counts()
    .sort_values(ascending=False)
    .plot(kind="bar")
)

plt.title("Distribution of Agent Roles")
plt.xlabel("Agent Role")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# Plot the average action risk score for each agent role.
avg_scores = (
    df.groupby("agent_role")["action_risk_score"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10, 5))

plt.plot(
    avg_scores.index,
    avg_scores.values,
    marker="o",
    linewidth=2
)

plt.title("Average Action Risk Score by Agent Role")
plt.xlabel("Agent Role")
plt.ylabel("Average Risk Score")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()