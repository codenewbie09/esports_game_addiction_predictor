import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/random_forest_model.pkl")

MODEL_PATH = os.path.normpath(MODEL_PATH)
model = joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Esports Game Addiction Predictor",
    page_icon="🎮",
    layout="centered",
)

st.title("🎮 Esports Game Addiction Predictor")
st.markdown("""
Predict the **addiction level** of a Steam game based on its price, ratings, and gameplay characteristics.
""")

st.markdown("---")


st.sidebar.header("🕹️ Input Game Details")

price = st.sidebar.number_input(
    "💰 Price (in USD)", min_value=0.0, max_value=120.0, value=10.0, step=0.5
)
positive_ratings = st.sidebar.number_input(
    "👍 Positive Ratings", min_value=0, max_value=3000000, value=500, step=50
)
negative_ratings = st.sidebar.number_input(
    "👎 Negative Ratings", min_value=0, max_value=500000, value=100, step=10
)
is_multiplayer = st.sidebar.selectbox(
    "🎯 Multiplayer Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
description_length = st.sidebar.slider(
    "📝 Description Length (characters)", 0, 400, 200
)
tag_count = st.sidebar.slider("🏷️ Total Tags", 1, 25, 10)
addictive_tag_count = st.sidebar.slider("🔥 Addictive Tags", 0, 10, 3)

price_log = np.log1p(price)
pos_log = np.log1p(positive_ratings)
neg_log = np.log1p(negative_ratings)

with st.sidebar.expander("ℹ️ What are 'Addictive Tags'?"):
    st.markdown("""
    Addictive tags are specific **game genres or features** that are commonly linked to **high player engagement** and **habit-forming gameplay**.
    These tags are derived from behavioral game design patterns observed in the Steam dataset.

    ### 🎮 Addictive Tags Used by the Model:
    ```
    multiplayer, co_op, pvp, team_based, battle_royale, e_sports,
    rpg, open_world, sandbox, survival, action, fps, strategy,
    simulation, casual, rogue_like, hack_and_slash, replay_value,
    progression, exploration
    ```

    ### 🧠 Why These Matter:
    - 🧩 **Multiplayer / Co-op / PvP / Team-Based / Battle Royale / eSports:** Encourage social competition and repeat play.
    - ⚔️ **Action / FPS / Hack & Slash:** Fast-paced feedback loops keep players engaged.
    - 🏕️ **RPG / Open World / Sandbox / Exploration:** Immersive experiences promote long-term investment.
    - 🔁 **Survival / Strategy / Progression / Replay Value:** Reward incremental progress and grind-based play.
    - 🎯 **Simulation / Casual / Rogue-Like:** Accessible but repetitive play patterns that can lead to habit formation.

    The model counts how many of these tags are associated with each game (`addictive_tag_count`) to estimate its **addictive potential**.
    More of these tags = higher likelihood of player addiction behavior.
    """)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Predict Addiction Level")

if predict_btn:
    # Prepare input dataframe (with internal log values)
    feature_names = [
        "price",
        "positive_ratings",
        "negative_ratings",
        "is_multiplayer",
        "description_length",
        "tag_count",
        "addictive_tag_count",
    ]

    input_data = pd.DataFrame(
        [
            [
                price_log,
                pos_log,
                neg_log,
                is_multiplayer,
                description_length,
                tag_count,
                addictive_tag_count,
            ]
        ],
        columns=feature_names,
    )

    prediction = model.predict(input_data)[0]
    addiction_map = {0: "Low", 1: "Medium", 2: "High"}
    addiction_label = addiction_map.get(prediction, "Unknown")

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if addiction_label == "Low":
        st.success(f"🟢 Predicted Addiction Level: **{addiction_label}**")
        st.write("This game shows low engagement and minimal addictive behavior.")
    elif addiction_label == "Medium":
        st.warning(f"🟡 Predicted Addiction Level: **{addiction_label}**")
        st.write(
            "This game exhibits moderate addictive potential with balanced engagement."
        )
    else:
        st.error(f"🔴 Predicted Addiction Level: **{addiction_label}**")
        st.write(
            "This game displays high addictive tendencies — strong engagement signals detected."
        )

    st.markdown("---")
    st.subheader("📊 Feature Importance (from Random Forest)")

    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=True)

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax)
    plt.title("Top Features Influencing Addiction Prediction", fontsize=12)
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    st.pyplot(fig)

    st.markdown("---")
    st.caption("Model: Random Forest | Developed by Prateek Agrawal (FDS Project)")

else:
    st.info(
        "👈 Adjust the game parameters in the sidebar and click **Predict Addiction Level** to begin."
    )
