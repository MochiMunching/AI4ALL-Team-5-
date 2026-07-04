from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import kagglehub

# load env variables
load_dotenv()

# Download latest version
path = kagglehub.dataset_download("algozee/agentic-ai-security-risk-dataset")

print("Path to dataset files:", path)
print(os.listdir(path))

# load dataset
csv_file = os.path.join(path, "agent_security_risk_scores.csv")
df = pd.read_csv(csv_file)

print(df.head())
print(df.columns)

# exploratory data analysis
def get_summary_stats(df):
  return df.describe()

def check_missing_values(df):
  return df.isnull().sum()

def find_extreme_outliers(df, col):
  return df[df[col] > df[col].quantile(0.95)]


print(get_summary_stats(df))
print(f'\n{check_missing_values(df)}')
outliers = find_extreme_outliers(df, 'action_risk_score')
print(f'\nDetected outliers: {outliers}')


# checking for bias
def check_group_representation(df, group_col):
    return df[group_col].value_counts(normalize=True)

def compare_scores_by_agent_role(df):
    return df.groupby('agent_role')['action_risk_score'].mean()

print(check_group_representation(df, 'agent_role'))
print(f'\n{compare_scores_by_agent_role(df)}')