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

# Banner section
st.markdown(
    """
    <h1 style="color: #ffffff; font-family: Arial, sans-serif; text-align: center;">💎 Diamond Brothers 💎</h1>
    <h3 style="color: #ffffff; font-family: Arial, sans-serif; text-align: center;">The world's best diamond pricing tool for (cat)boosting transparency and confidence!</h3>
    """,
    unsafe_allow_html=True
)

# Sidebar welcome message
st.sidebar.header("Welcome to the home of Diamond pricing!")
st.sidebar.write("""
This Diamond Brothers app gives you a clear concise and digestible overview of diamond pricing mechanics! 
The app uses a CatBoost model to predict the price of any diamond based on: Carat, Color, Cut, and Clarity. 
Have fun and keep on learning!
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
    st.image("Diamond_Carat_Weight.png", caption="Comparison of Diamond Sizes (Carats)", use_container_width=True)

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
    import streamlit as st
    import pandas as pd
    from catboost import CatBoostRegressor
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score
    
    st.header("Diamond Price Prediction Tool")
    st.write(
        "This section predicts diamond prices using a tuned CatBoost model. "
        "You can also compare the performance of pre-tuned and tuned models below."
    )
    
    # Define feature list explicitly
    FEATURE_COLUMNS = ['Carat', 'Cut', 'Color', 'Clarity', 'Depth', 'Table', 'X', 'Y', 'Z']
    
    # Load pre-trained models
    @st.cache_resource
    def load_models():
        try:
            pre_tuned_model = CatBoostRegressor()
            tuned_model = CatBoostRegressor()
            pre_tuned_model.load_model("pre_tuned_catboost_model.cbm")
            tuned_model.load_model("tuned_catboost_model.cbm")
            return pre_tuned_model, tuned_model
        except FileNotFoundError:
            st.error("Model files not found. Ensure 'pre_tuned_catboost_model.cbm' and 'tuned_catboost_model.cbm' exist.")
            st.stop()
    
    pre_tuned_model, tuned_model = load_models()
    
    # Load dataset
    @st.cache_data()
    def load_data():
        df = pd.read_csv("diamonds.csv")
        df.columns = df.columns.str.capitalize()
        return df.dropna()
    
    df = load_data()
    
    # Calculate default values for Depth, Table, X, Y, Z
    default_depth = df["Depth"].mean()
    default_table = df["Table"].mean()
    default_x = df["X"].mean()
    default_y = df["Y"].mean()
    default_z = df["Z"].mean()
    
    # Prediction Section
    st.subheader("Make Predictions")
    with st.form("prediction_form"):
        Carat = st.slider("Carat", min_value=float(df["Carat"].min()), max_value=float(df["Carat"].max()), value=1.0, step=0.01)
        Cut = st.selectbox("Cut", options=df["Cut"].unique())
        Color = st.selectbox("Color", options=df["Color"].unique())
        Clarity = st.selectbox("Clarity", options=df["Clarity"].unique())
        submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'Carat': [Carat],
            'Cut': [Cut],
            'Color': [Color],
            'Clarity': [Clarity],
            'Depth': [default_depth],  # Add default depth
            'Table': [default_table],  # Add default table
            'X': [default_x],          # Add default X
            'Y': [default_y],          # Add default Y
            'Z': [default_z]           # Add default Z
        })
    
        # Ensure input matches training feature set
        input_data = input_data[FEATURE_COLUMNS]
    
        # Predict using the tuned model
        try:
            tuned_prediction = tuned_model.predict(input_data)[0]
    
            # Display prediction
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h2>Predicted Price:</h2>
                <h1>${tuned_prediction:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
    
    # Compare Pre-Tuned and Tuned Model Performance
    st.subheader("Model Performance Comparison")
    with st.expander("View Pre-Tuned vs Tuned Model Metrics"):
        # Prepare features and target for evaluation
        X = df[FEATURE_COLUMNS]
        y = df['Price']
    
        # Evaluate pre-tuned model
        pre_tuned_predictions = pre_tuned_model.predict(X)
        pre_rmse = np.sqrt(mean_squared_error(y, pre_tuned_predictions))
        pre_r2 = r2_score(y, pre_tuned_predictions)
    
        # Evaluate tuned model
        tuned_predictions = tuned_model.predict(X)
        tuned_rmse = np.sqrt(mean_squared_error(y, tuned_predictions))
        tuned_r2 = r2_score(y, tuned_predictions)
    
        # Display metrics
        st.markdown(f"""
        **Pre-Tuned Model Performance:**
        - RMSE: {pre_rmse:.2f}
        - R²: {pre_r2:.2f}
    
        **Tuned Model Performance:**
        - RMSE: {tuned_rmse:.2f}
        - R²: {tuned_r2:.2f}
        """)



with tab4:
    st.header("Statistical Visualization")
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Carat and Cut Distributions (Side by Side)
    col1, col2 = st.columns(2)

    with col1:
        # Distribution of Carat
        fig1, ax1 = plt.subplots(figsize=(4, 2))
        sns.histplot(df['Carat'], bins=50, kde=True, color='#739BD0', ax=ax1)
        ax1.set_title('Distribution of Carat', fontsize=14)
        ax1.set_xlabel('Carat', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig1)

    with col2:
        # Distribution of Cut
        fig2, ax2 = plt.subplots(figsize=(4, 2))
        sns.countplot(data=df, x='Cut', palette=['#739BD0'], ax=ax2)
        ax2.set_title('Distribution of Cut', fontsize=14)
        ax2.set_xlabel('Cut', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig2)

    # Color and Clarity Distributions (Side by Side)
    col3, col4 = st.columns(2)

    with col3:
        # Distribution of Color
        fig3, ax3 = plt.subplots(figsize=(4, 2))
        sns.countplot(data=df, x='Color', palette=['#739BD0'], ax=ax3)
        ax3.set_title('Distribution of Color', fontsize=14)
        ax3.set_xlabel('Color', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig3)

    with col4:
        # Distribution of Clarity
        fig4, ax4 = plt.subplots(figsize=(4, 2))
        sns.countplot(data=df, x='Clarity', palette=['#739BD0'], ax=ax4)
        ax4.set_title('Distribution of Clarity', fontsize=14)
        ax4.set_xlabel('Clarity', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig4)

    # Hexbin plot: Price vs Carat
    col5, col6 = st.columns([2, 1])  # Adjust column widths for balance

    with col5:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        hb = ax5.hexbin(df['Carat'], df['Price'], gridsize=50, cmap='viridis', mincnt=1)  # Hexbin doesn't support single color
        cb = plt.colorbar(hb, ax=ax5, label='Count')
        ax5.set_title('Price vs Carat', fontsize=14)
        ax5.set_xlabel('Carat', fontsize=12)
        ax5.set_ylabel('Price ($)', fontsize=12)
        st.pyplot(fig5)

    with col6:
        # Feature Importance Visualization
        st.subheader("Feature Importance")

        # Calculate feature importance from tuned model
        feature_importances = tuned_model.get_feature_importance(prettified=True)
        feature_importances_df = pd.DataFrame(feature_importances, columns=['Feature', 'Importance'])
        feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        fig6, ax6 = plt.subplots(figsize=(4, 4))
        sns.barplot(x='Importance', y='Feature', data=feature_importances_df, palette=['#739BD0'], ax=ax6)
        ax6.set_title('Feature Importance', fontsize=14)
        ax6.set_xlabel('Importance', fontsize=12)
        ax6.set_ylabel('Feature', fontsize=12)
        st.pyplot(fig6)
