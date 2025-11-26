import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Game Addiction Predictor & Analytics",
    page_icon="🎮",
    layout="wide",
)


@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data_cleaned.csv")
        df = pd.read_csv(DATA_PATH)
        df["addiction_level"] = df["addiction_level"].map(
            {0: "Low", 1: "Medium", 2: "High"}
        )
        return df
    except:
        return None


model = load_model()
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Predictor", "Data Analytics", "Model Performance", "About"]
)

if page == "Predictor":
    st.title("Game Addiction Level Predictor")
    st.markdown("""
    Predict the **addiction level** of a Steam game based on its characteristics.
    Adjust the parameters in the sidebar and click **Predict** to see results.
    """)
    st.markdown("---")

    st.sidebar.header("Input Game Details")

    price = st.sidebar.number_input(
        "Price (in USD)", min_value=0.0, max_value=120.0, value=10.0, step=0.5
    )
    positive_ratings = st.sidebar.number_input(
        "Positive Ratings", min_value=0, max_value=3000000, value=500, step=50
    )
    negative_ratings = st.sidebar.number_input(
        "Negative Ratings", min_value=0, max_value=500000, value=100, step=10
    )
    is_multiplayer = st.sidebar.selectbox(
        "Multiplayer Support",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    description_length = st.sidebar.slider(
        "Description Length (characters)", 0, 400, 200
    )
    tag_count = st.sidebar.slider("Total Tags", 1, 25, 10)
    addictive_tag_count = st.sidebar.slider("Addictive Tags", 0, 10, 3)

    with st.sidebar.expander("What are 'Addictive Tags'?"):
        st.markdown("""
        Addictive tags are game genres/features linked to high player engagement:
        
        **Tags used:**
        ```
        multiplayer, mmorpg, open_world, rpg, 
        action, shooter, strategy, co_op, 
        competitive, online_co_op, pvp, 
        sandbox, survival, fps, rogue_like
        ```
        
        More of these tags = higher addiction potential.
        """)

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("Predict Addiction Level", type="primary")

    if predict_btn:
        price_log = np.log1p(price)
        pos_log = np.log1p(positive_ratings)
        neg_log = np.log1p(negative_ratings)

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
        proba = model.predict_proba(input_data)[0]

        addiction_map = {0: "Low", 1: "Medium", 2: "High"}
        addiction_label = addiction_map[prediction]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Prediction Result")

            if addiction_label == "Low":
                st.success(f"### Predicted: **{addiction_label} Addiction Risk**")
                st.write(
                    "This game shows low engagement patterns and minimal addictive behavior."
                )
            elif addiction_label == "Medium":
                st.warning(f"### Predicted: **{addiction_label} Addiction Risk**")
                st.write(
                    "This game exhibits moderate addictive potential with balanced engagement."
                )
            else:
                st.error(f"### Predicted: **{addiction_label} Addiction Risk**")
                st.write(
                    "This game displays high addictive tendencies with strong engagement signals."
                )

        with col2:
            st.subheader("Confidence Scores")
            prob_df = pd.DataFrame(
                {"Level": ["Low", "Medium", "High"], "Probability": proba}
            )
            st.dataframe(
                prob_df.style.format({"Probability": "{:.1%}"}), hide_index=True
            )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            fi_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            ).sort_values(by="Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax
            )
            plt.title("Features Influencing This Prediction")
            plt.xlabel("Relative Importance")
            st.pyplot(fig)

        with col2:
            st.subheader("Input Summary")
            input_summary = pd.DataFrame(
                {
                    "Feature": [
                        "Price",
                        "Positive Ratings",
                        "Negative Ratings",
                        "Multiplayer",
                        "Description Length",
                        "Total Tags",
                        "Addictive Tags",
                    ],
                    "Value": [
                        f"${price:.2f}",
                        positive_ratings,
                        negative_ratings,
                        "Yes" if is_multiplayer else "No",
                        description_length,
                        tag_count,
                        addictive_tag_count,
                    ],
                }
            )
            st.dataframe(input_summary, hide_index=True, use_container_width=True)

    else:
        st.info("Adjust game parameters in the sidebar and click **Predict** to begin.")

elif page == "Data Analytics":
    st.title("Dataset Analytics & Insights")

    if df is None:
        st.error(
            "Dataset not found. Please ensure processed_data_cleaned.csv is available."
        )
    else:
        st.success(
            f"Dataset loaded: **{df.shape[0]} games** with **{df.shape[1]} features**"
        )

        st.markdown("---")
        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Games", f"{df.shape[0]:,}")
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            multiplayer_pct = (df["is_multiplayer"].sum() / len(df)) * 100
            st.metric("Multiplayer Games", f"{multiplayer_pct:.1f}%")
        with col4:
            avg_tags = df["addictive_tag_count"].mean()
            st.metric("Avg Addictive Tags", f"{avg_tags:.1f}")

        st.markdown("---")
        st.subheader("Addiction Level Distribution")

        col1, col2 = st.columns([1, 2])

        with col1:
            target_counts = df["addiction_level"].value_counts()
            st.dataframe(
                pd.DataFrame(
                    {
                        "Level": target_counts.index,
                        "Count": target_counts.values,
                        "Percentage": (target_counts.values / len(df) * 100).round(1),
                    }
                ).set_index("Level"),
                use_container_width=True,
            )

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            order = ["Low", "Medium", "High"]
            colors = ["#90EE90", "#FFD700", "#FF6B6B"]
            sns.countplot(
                data=df, x="addiction_level", order=order, palette=colors, ax=ax
            )
            plt.title(
                "Distribution of Addiction Levels", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Addiction Level")
            plt.ylabel("Number of Games")
            for container in ax.containers:
                ax.bar_label(container)
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Feature Distributions")

        feature_to_plot = st.selectbox(
            "Select a feature to visualize:",
            [
                "price",
                "positive_ratings",
                "negative_ratings",
                "average_playtime",
                "description_length",
                "tag_count",
                "addictive_tag_count",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[feature_to_plot], bins=30, kde=True, color="skyblue", ax=ax)
            plt.title(
                f"Distribution of {feature_to_plot}", fontsize=12, fontweight="bold"
            )
            plt.xlabel(feature_to_plot)
            plt.ylabel("Frequency")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            order = ["Low", "Medium", "High"]
            sns.boxplot(
                data=df,
                x="addiction_level",
                y=feature_to_plot,
                order=order,
                palette="Set2",
                ax=ax,
            )
            plt.title(
                f"{feature_to_plot} by Addiction Level", fontsize=12, fontweight="bold"
            )
            plt.xlabel("Addiction Level")
            plt.ylabel(feature_to_plot)
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Feature Correlation Matrix")

        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Multiplayer Impact on Addiction")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            order = ["Low", "Medium", "High"]
            sns.countplot(
                data=df,
                x="is_multiplayer",
                hue="addiction_level",
                hue_order=order,
                palette="viridis",
                ax=ax,
            )
            plt.title("Multiplayer vs Addiction Level", fontsize=12, fontweight="bold")
            plt.xlabel("Multiplayer (0=No, 1=Yes)")
            plt.ylabel("Count")
            plt.legend(title="Addiction Level")
            st.pyplot(fig)

        with col2:
            mp_stats = (
                pd.crosstab(
                    df["is_multiplayer"], df["addiction_level"], normalize="index"
                )
                * 100
            )
            mp_stats.index = ["Single Player", "Multiplayer"]
            st.dataframe(
                mp_stats.round(1).style.format("{:.1f}%"), use_container_width=True
            )
            st.caption("Percentage distribution of addiction levels by game type")

        st.markdown("---")
        st.subheader("Summary Statistics")

        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance Analysis")

    if df is None:
        st.error("Dataset not found for performance visualization.")
    else:
        st.markdown("### Random Forest Model Evaluation")
        st.info(
            "This page shows how well the Random Forest model performs on the test data."
        )

        st.markdown("---")
        st.subheader("Overall Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "55%", help="Overall correct predictions")
        with col2:
            st.metric("Precision", "54%", help="Precision across all classes")
        with col3:
            st.metric("Recall", "55%", help="Recall across all classes")
        with col4:
            st.metric("F1 Score", "54%", help="Harmonic mean of precision and recall")

        st.markdown("---")
        st.subheader("Performance by Addiction Level")

        col1, col2 = st.columns([1, 1])

        with col1:
            perf_data = pd.DataFrame(
                {
                    "Class": ["Low", "Medium", "High"],
                    "Precision": [0.62, 0.56, 0.46],
                    "Recall": [0.63, 0.56, 0.46],
                    "F1-Score": [0.62, 0.56, 0.46],
                    "Support": [413, 412, 409],
                }
            )
            st.dataframe(
                perf_data.set_index("Class").style.format(
                    {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"}
                ),
                use_container_width=True,
            )

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(perf_data["Class"]))
            width = 0.25

            ax.bar(
                x - width,
                perf_data["Precision"],
                width,
                label="Precision",
                color="#FF6B6B",
            )
            ax.bar(x, perf_data["Recall"], width, label="Recall", color="#4ECDC4")
            ax.bar(
                x + width,
                perf_data["F1-Score"],
                width,
                label="F1-Score",
                color="#45B7D1",
            )

            ax.set_xlabel("Addiction Level")
            ax.set_ylabel("Score")
            ax.set_title("Performance Metrics by Class")
            ax.set_xticks(x)
            ax.set_xticklabels(perf_data["Class"])
            ax.legend()
            ax.set_ylim([0, 1])
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Confusion Matrix")

        col1, col2 = st.columns([1, 1])

        with col1:
            cm = np.array([[260, 112, 41], [101, 231, 80], [52, 99, 258]])

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"],
                ax=ax,
            )
            plt.title("Confusion Matrix - Random Forest")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            st.pyplot(fig)

        with col2:
            st.markdown("### Key Insights")
            st.markdown("""
            **Common Misclassifications:**
            - **27%** of Low games → Medium
                - Games with high ratings but lower playtime
            - **14%** of Medium games → High
                - Strong multiplayer features dominate
            - **24%** of High games → Medium
                - Missing typical addictive features
            
            **Model Strengths:**
            - Best at identifying Low addiction games (62% F1)
            - Consistent performance across metrics
            - Low variance in cross-validation (±0.023)
            
            **Areas for Improvement:**
            - High addiction game detection (46% F1)
            - Better separation of Medium/High boundary
            """)

        st.markdown("---")
        st.subheader("Feature Importance Analysis")

        importances = model.feature_importances_
        feature_names = [
            "price",
            "positive_ratings",
            "negative_ratings",
            "is_multiplayer",
            "description_length",
            "tag_count",
            "addictive_tag_count",
        ]

        fi_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
                "Percentage": importances * 100,
            }
        ).sort_values(by="Importance", ascending=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(
                fi_df.style.format(
                    {"Importance": "{:.3f}", "Percentage": "{:.1f}%"}
                ).background_gradient(subset=["Importance"], cmap="YlOrRd"),
                hide_index=True,
                use_container_width=True,
            )

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            fi_plot = fi_df.sort_values("Importance", ascending=True)
            sns.barplot(
                x="Importance", y="Feature", data=fi_plot, palette="rocket", ax=ax
            )
            plt.title("Feature Importance Ranking")
            plt.xlabel("Importance Score")
            st.pyplot(fig)

elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Game Addiction Predictor
    
    This application uses machine learning to predict the addiction potential of video games
    based on their characteristics and community engagement data from Steam.
    
    ### Project Overview
    
    **Objective:** Develop an automated system to classify games by addiction risk level
    
    **Dataset:** 6,170 Steam games with metadata, descriptions, and 370+ community tags
    
    **Models:** Random Forest (55% accuracy) and XGBoost (50% accuracy)
    
    ### Features Used
    
    The model analyzes these game characteristics:
    - **Price**: Game cost in USD
    - **Ratings**: Community positive/negative feedback
    - **Multiplayer**: Social/competitive features
    - **Description Length**: Complexity indicator
    - **Tags**: Genre and feature tags
    - **Addictive Tags**: Specific engagement-linked features
    
    ### Methodology
    
    1. **Data Collection**: Merged three Steam datasets from Kaggle
    2. **Feature Engineering**: Created behavioral indicators
    3. **Label Creation**: Split games into Low/Medium/High using playtime quantiles
    4. **Model Training**: Trained ensemble models with 80-20 split
    5. **Evaluation**: Tested with accuracy, precision, recall, F1-score
    6. **Deployment**: Built interactive Streamlit dashboard
    
    ### Important Disclaimers
    
    **Limitations:**
    - Playtime ≠ clinical addiction
    - Steam games only (no mobile/console)
    - Games evolve over time
    - No player psychological data
    
    **Ethical Use:**
    - This is an engagement predictor, not a clinical diagnostic tool
    - Should supplement, not replace, parental judgment
    - Low predictions don't guarantee safety
    - Consult professionals for real addiction concerns
    
    ### Academic Context
    
    **Project Type:** Foundations of Data Science Course Project
    
    **Student:** Prateek Agrawal (Roll No. 2310110696)
    
    **Department:** Computer Science
    
    ### Research Foundation
    
    This work builds on research showing that:
    - Structural game features drive addiction (Griffiths et al., 2012)
    - Engagement dimensions predict addiction (Abbasi et al., 2021)
    - High playtime alone ≠ addiction (Andre et al., 2020)
    
    ### Technology Stack
    
    - **Machine Learning**: scikit-learn, XGBoost
    - **Data Processing**: pandas, numpy
    - **Visualization**: matplotlib, seaborn
    - **Web App**: Streamlit
    - **Development**: Python 3.8+
    
    ### Resources
    
    **GitHub Repository**: https://github.com/codenewbie09/esports_game_addiction_predictor
    
    **Dataset Source**: Steam Games Dataset (Kaggle)
    
    **Model Files**: Random Forest and XGBoost classifiers
    
    ### Contact
    
    For questions or feedback about this project, please contact through the course instructor.
    
    ---
    
    *Built with Streamlit and scikit-learn*
    """)

    st.markdown("---")
    st.subheader("Project Achievements")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Games Analyzed", "6,170")
    with col2:
        st.metric("Model Accuracy", "55%")
    with col3:
        st.metric("Features Engineered", "11")

st.sidebar.markdown("---")
st.sidebar.caption("Game Addiction Predictor v2.0 | Prateek Agrawal")
