#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:53:49 2024

@author: clemensburleson
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#### Defining general properties of the app
###########################################
st.set_page_config(
    page_title="Diamond Brothers",
    page_icon='💎',
    layout='wide'
)

# Apply custom CSS for dark background
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the banner section
st.markdown(
    """
    <div style="background-color: #000000; padding: 20px; text-align: center; border-radius: 5px; margin-bottom: 20px;">
        <h1 style="color: #ffffff; font-family: Arial, sans-serif;">💎 Welcome to Diamond Brothers 💎</h1>
        <h3 style="color: #ffffff; font-family: Arial, sans-serif;">The world's best diamond pricing tool for transparency and confidence</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar welcome message
st.sidebar.header("Welcome!")
st.sidebar.write("""
This Diamond Brothers app uses advanced machine learning techniques to predict diamond prices based on attributes such as carat, cut, clarity, and color. 
Customize your preferences, and view the filtered data, to make informed pricing decisions!
""")

st.markdown("By Clemens Burleson & Aksh Iyer from the University of St. Gallen under the instruction of Prof. Dr. Ivo Blohm")

#### Define Load functions and load data
###########################################
@st.cache_data()
def load_data():
    # Load the data and rename columns to have capitalized first letters
    df = pd.read_csv("diamonds.csv")
    df.columns = df.columns.str.capitalize()  # Capitalize all column titles
    return df.dropna()

df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Diamond Guide", "Filtered Diamonds", "Price Prediction", "Pricing Relationships"])

with tab1:
    st.header("Diamond Color Guide")
    # Dictionary to map diamond colors to visual representations
    color_descriptions = {
        "D": "Completely colorless, the highest grade of diamond color.",
        "E": "Exceptional white, minute traces of color detectable.",
        "F": "Excellent white, slight color detectable only by experts.",
        "G": "Near colorless, slight warmth noticeable.",
        "H": "Near colorless, slightly more warmth.",
        "I": "Noticeable warmth, light yellow tint visible.",
        "J": "Very noticeable warmth, visible yellow tint.",
    }

    # Create colored boxes to represent diamond colors
    diamond_color_boxes = {
        "D": "#fdfdfd",  # White
        "E": "#f8f8f8",  # Slightly off-white
        "F": "#f0f0f0",  # Slight gray
        "G": "#e8e8e8",  # Light gray
        "H": "#e0d4b8",  # Slight yellow
        "I": "#d4b892",  # Yellow tint
        "J": "#ccb78e",  # Warm yellow
    }

    # Display diamond colors with descriptions
    for color, hex_code in diamond_color_boxes.items():
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 50px; height: 50px; background-color: {hex_code}; border: 1px solid black; margin-right: 15px;"></div>
                <div>
                    <strong>Color {color}:</strong> {color_descriptions[color]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Add a section for Diamond Sizes
    st.header("Diamond Sizes (Carats)")

    # Display the image with a caption
    st.image("Diamond_Carat_Weight.png", caption="Comparison of Diamond Sizes (Carats)", use_column_width=True)

with tab2:
    st.header("Filtered Diamonds")

    # Create two main columns for layout
    col1, col2 = st.columns([1, 1])  # Equal-width columns for filter options and filtered data

    with col1:  # Filter options on the left
        st.subheader("Filter Options")

        # Slider for price range (formatted with commas, no decimals)
        price_range = st.slider(
            "Select Desired Price Range",
            min_value=int(df["Price"].min()),  # Cast to int to remove decimals
            max_value=int(df["Price"].max()),  # Cast to int to remove decimals
            value=(int(df["Price"].min()), int(df["Price"].max())),  # Set initial range as integers
            format="%d"  # Display numbers without decimals
        )
    
        # Slider for carat range
        mass_range = st.slider(
            "Select Desired Carat Range",
            min_value=float(df["Carat"].min()),
            max_value=float(df["Carat"].max()),
            value=(float(df["Carat"].min()), float(df["Carat"].max()))
        )
    
        # Multiselect options for Cut, Color, and Clarity
        cut_options = st.multiselect(
            "Select Diamond Cuts",
            options=df["Cut"].unique(),
            default=df["Cut"].unique()
        )
    
        color_options = st.multiselect(
            "Select Diamond Colors",
            options=df["Color"].unique(),
            default=df["Color"].unique()
        )
    
        clarity_options = st.multiselect(
            "Select Diamond Clarity Levels",
            options=df["Clarity"].unique(),
            default=df["Clarity"].unique()
        )
    
        # Multiselect for column selection
        st.subheader("Customize Columns")
        default_columns = ['Price', 'Carat', 'Cut', 'Color']  # Default columns to display
        columns_to_display = st.multiselect(
            "Select Columns to Display:",
            options=df.columns.tolist(),
            default=[col for col in default_columns if col in df.columns]  # Use default columns if available
        )

    with col2:  # Filtered data on the right
        st.subheader("Filtered Diamonds")

        # Apply filters to the DataFrame
        filtered_diamonds = df[
            (df["Price"] >= price_range[0]) &
            (df["Price"] <= price_range[1]) &
            (df["Carat"] >= mass_range[0]) &
            (df["Carat"] <= mass_range[1]) &
            (df["Cut"].isin(cut_options)) &
            (df["Color"].isin(color_options)) &
            (df["Clarity"].isin(clarity_options))
        ]

        num_results = len(filtered_diamonds)
        st.markdown(f"**{num_results} results**")

        if filtered_diamonds.empty:
            st.warning("No diamonds match your selected criteria. Please adjust the filters.")
        else:
            # Center the filtered data horizontally
            st.markdown(
                """
                <div style="display: flex; justify-content: center; width: 100%; margin-top: 20px;">
                    <div style="width: 70%; max-width: 800px;">
                """,
                unsafe_allow_html=True
            )
            # Display the filtered DataFrame with selected columns
            st.dataframe(filtered_diamonds[columns_to_display].reset_index(drop=True))  # Reset index and drop the original one
            st.markdown(
                """
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
from sklearn.ensemble import GradientBoostingRegressor

with tab3:
    st.header("Diamond Price Prediction Tool")
    st.write("Train and evaluate a Gradient Boosting model on the diamond dataset. Compare pre-tuned and tuned model performance.")

    # Split data into features and target
    X = df.drop(columns=['Price', 'Table', 'Depth'])  # Drop unused columns
    y = df['Price']  # Target variable

    # Encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_features:
        X[col] = X[col].astype('category').cat.codes
    
    from sklearn.model_selection import train_test_split
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train pre-tuned model
    @st.cache_resource
    def train_pre_tuned_model(X_train, y_train, X_test, y_test):
        pre_tuned_model = GradientBoostingRegressor(random_state=42)
        pre_tuned_model.fit(X_train, y_train)
        pre_preds = pre_tuned_model.predict(X_test)
        pre_rmse = np.sqrt(mean_squared_error(y_test, pre_preds))
        pre_r2 = r2_score(y_test, pre_preds)
        return pre_tuned_model, pre_rmse, pre_r2

    # Train tuned model
    @st.cache_resource
    def train_tuned_model(X_train, y_train, X_test, y_test):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.01],
            'subsample': [0.8, 1.0]
        }
        grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Best tuned model
        tuned_model = grid_search.best_estimator_
        tuned_preds = tuned_model.predict(X_test)
        tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
        tuned_r2 = r2_score(y_test, tuned_preds)
        return tuned_model, tuned_rmse, tuned_r2

    # Train both models and retrieve performance metrics
    pre_tuned_model, pre_rmse, pre_r2 = train_pre_tuned_model(X_train, y_train, X_test, y_test)
    tuned_model, tuned_rmse, tuned_r2 = train_tuned_model(X_train, y_train, X_test, y_test)

    # Display metrics
    st.subheader("Model Performance Comparison")
    st.write("Evaluate the performance of the pre-tuned and tuned models:")
    st.markdown(f"""
    **Pre-Tuned Model Performance:**
    - RMSE: {pre_rmse:.2f}
    - R²: {pre_r2:.2f}

    **Tuned Model Performance:**
    - RMSE: {tuned_rmse:.2f}
    - R²: {tuned_r2:.2f}
    """)

    # Prediction Section
    st.subheader("Make Predictions")

    # Input fields
    with st.form("prediction_form"):
        Carat = st.slider("Carat", min_value=float(df["Carat"].min()), max_value=float(df["Carat"].max()), value=1.0, step=0.01)
        Cut = st.selectbox("Cut", options=df["Cut"].unique())
        Color = st.selectbox("Color", options=df["Color"].unique())
        Clarity = st.selectbox("Clarity", options=df["Clarity"].unique())
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Carat': [Carat],
            'Cut': [Cut],
            'Color': [Color],
            'Clarity': [Clarity]
        })

        # Encode categorical features for the model
        for col in categorical_features:
            input_data[col] = input_data[col].astype('category').cat.codes

        # Predict using the tuned model
        prediction = tuned_model.predict(input_data)[0]

        # Display the prediction
        st.markdown(
            f"""
            <div style="background-color: #739BD0; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                <h2 style="color: #ffffff;">Estimated Price:</h2>
                <h1 style="color: #ffffff;">${prediction:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Collapsible metrics section
    with st.expander("Model Performance Details"):
        st.write("#### Performance Comparison")
        st.markdown(f"""
        - **Pre-Tuned Model RMSE:** {pre_rmse:.2f}, R²: {pre_r2:.2f}
        - **Tuned Model RMSE:** {tuned_rmse:.2f}, R²: {tuned_r2:.2f}
        """)


with tab4:
    st.header("Pricing Relationships")
    if 'filtered_diamonds' in locals() and not filtered_diamonds.empty:
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_diamonds, x="Carat", y="Price", hue="Color", palette="viridis", ax=ax)
        ax.set_title("Carat vs. Price", fontsize=16)
        ax.set_xlabel("Carat", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected filters.")
