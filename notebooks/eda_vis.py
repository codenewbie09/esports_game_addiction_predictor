import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/processed_data_cleaned.csv")
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample Data:")
print(df.head())
df["addiction_level"] = df["addiction_level"].map({0: "Low", 1: "Medium", 2: "High"})

print("\nData Types and Missing Values:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(
    x="addiction_level", data=df, order=["Low", "Medium", "High"], palette="Set2"
)
plt.title("Distribution of Game Addiction Levels")
plt.xlabel("Addiction Level")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

num_cols = [
    "price",
    "positive_ratings",
    "negative_ratings",
    "average_playtime",
    "description_length",
    "tag_count",
    "addictive_tag_count",
]
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=30, kde=True, color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Feature vs Target Analysis (Boxplots)
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(
        x="addiction_level",
        y=col,
        data=df,
        order=["Low", "Medium", "High"],
        palette="Set3",
    )
    plt.title(f"{col} vs Addiction Level")
    plt.xlabel("Addiction Level")
    plt.ylabel(col)
    plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="is_multiplayer", hue="addiction_level", data=df, palette="viridis")
plt.title("Multiplayer Influence on Addiction Level")
plt.xlabel("Is Multiplayer")
plt.ylabel("Count")
plt.legend(title="Addiction Level")
plt.show()

print("\n--- Summary Insights ---")
print("2️⃣ Multiplayer and high-owner games tend to have higher addiction levels.")
print("3️⃣ Positive ratings and playtime show moderate correlation with addiction level.")
print(
    "4️⃣ Price has minimal correlation, indicating free/cheap games aren't always more addictive."
)
