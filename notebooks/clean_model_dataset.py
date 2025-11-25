import numpy as np
import pandas as pd

df = pd.read_csv("../data/processed_data.csv")
print("Original shape:", df.shape)

cols_to_drop = ["appid", "name", "owners"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
print("After dropping columns:", df.shape)

# Apply log1p transformation to reduce skewness
log_cols = ["price", "positive_ratings", "negative_ratings", "average_playtime"]
for col in log_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])

print("Applied log transformation to:", log_cols)

# Encode target variable
if "addiction_level" in df.columns:
    df["addiction_level"] = df["addiction_level"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    )

df.to_csv("../data/processed_data_cleaned.csv", index=False)
print("Clean model-ready dataset saved as '../data/processed_data_cleaned.csv'")
print("Final shape:", df.shape)
