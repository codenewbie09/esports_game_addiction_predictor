import pandas as pd

steam = pd.read_csv("../data/steam.csv")
desc = pd.read_csv("../data/steam_description_data.csv")
tags = pd.read_csv("../data/steamspy_tag_data.csv")

print("Loaded Datasets:")
print(f"steam.csv: {steam.shape}")
print(f"steam_description_data.csv: {desc.shape}")
print(f"steamspy_tag_data.csv: {tags.shape}")

steam = steam[
    [
        "appid",
        "name",
        "price",
        "positive_ratings",
        "negative_ratings",
        "owners",
        "average_playtime",
        "median_playtime",
        "categories",
        "genres",
    ]
].dropna()

# Convert owners from range string (e.g., "20,000 - 50,000") to average numeric value


def parse_owners(x):
    try:
        lo, hi = x.split(" - ")
        return (int(lo.replace(",", "")) + int(hi.replace(",", ""))) / 2
    except:
        return None


steam["owners"] = steam["owners"].apply(parse_owners)

# Extract multiplayer flag from categories
steam["is_multiplayer"] = steam["categories"].apply(
    lambda x: 1 if "Multiplayer" in str(x) else 0
)

desc = desc[["steam_appid", "short_description"]]
merged = steam.merge(desc, left_on="appid", right_on="steam_appid", how="left")

merged["description_length"] = merged["short_description"].fillna("").apply(len)

# Drop 'appid' column and treat the rest as tag weights (0 = not tagged, >0 = strength)
tag_cols = [c for c in tags.columns if c != "appid"]

# List of tags related to addictive or engaging gameplay
popular_tags = [
    "multiplayer",
    "mmorpg",
    "open_world",
    "rpg",
    "action",
    "shooter",
    "strategy",
    "co_op",
    "competitive",
    "online_co_op",
    "pvp",
    "sandbox",
    "survival",
    "fps",
    "rogue_like",
]


def tag_features(row):
    game_tags = tags.loc[tags["appid"] == row["appid"], tag_cols]
    if game_tags.empty:
        return pd.Series([0, 0])
    tag_row = game_tags.iloc[0]
    tag_count = (tag_row > 0).sum()
    addictive_tag_count = sum(
        tag_row.get(tag, 0) > 0 for tag in popular_tags if tag in tag_row.index
    )
    return pd.Series([tag_count, addictive_tag_count])


merged[["tag_count", "addictive_tag_count"]] = merged.apply(tag_features, axis=1)

# Create addiction level based on average_playtime
merged = merged.dropna(subset=["average_playtime"])
merged = merged[merged["average_playtime"] > 0]
merged["addiction_level"] = pd.qcut(
    merged["average_playtime"], q=3, labels=["Low", "Medium", "High"], duplicates="drop"
)

final_df = merged[
    [
        "appid",
        "name",
        "price",
        "positive_ratings",
        "negative_ratings",
        "owners",
        "average_playtime",
        "is_multiplayer",
        "description_length",
        "tag_count",
        "addictive_tag_count",
        "addiction_level",
    ]
]

final_df.to_csv("../data/processed_data.csv", index=False)
print(
    f"Processed dataset saved with {final_df.shape[0]} rows and {
        final_df.shape[1]
    } columns."
)
