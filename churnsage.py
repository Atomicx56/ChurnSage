import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



background_image = "https://i.imgur.com/FzT93Rm.jpeg"
logo_image = "E:/logo.png"

# Display logo at the top
st.image(logo_image, width=200)

# Preprocess the data
def preprocess_data(df):
    """Preprocess the data: impute missing values and handle categorical columns."""
    numerical_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    if 'seniorcitizen' in df.columns:
        df['seniorcitizen'] = df['seniorcitizen'].map({0: "no", 1: "yes"})
    return df

# Encode categorical columns
def encode_categorical_columns(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    ct = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )
    df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())
    return df

# Identify the churn column
def identify_churn_column(df):
    potential_columns = ['Churn', 'Exited', 'Target', 'Attrition', 'Churned', 'IsChurn', 'customer_churn']
    for column in df.columns:
        if column in potential_columns:
            return column
        unique_values = df[column].dropna().unique()
        if set(unique_values) == {0, 1} or set(unique_values) == {'Yes', 'No'}:
            return column
    return None

# Train and evaluate the model
def train_and_evaluate_model(df, selected_features, target_column):
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 6, 8, 10, 12],
        'criterion': ['gini', 'entropy'],
    }

    grid_search = RandomizedSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_iter=15, n_jobs=-1, random_state=42
    )

    X = df[selected_features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    importances = best_model.feature_importances_

    return best_model, test_accuracy, test_predictions, X_test, y_test, importances

# Plot churn rate
def plot_churn_rate(df, churn_column):
    churn_values = df[churn_column].value_counts()
    churn_labels = churn_values.index.tolist()

    fig = go.Figure(data=[go.Pie(
        labels=churn_labels,
        values=churn_values,
        hole=.3,
        marker_colors=['#66b3ff', '#ff9999'],
        textinfo='percent+label'
    )])

    fig.update_layout(
        title_text="Churn Rate Distribution",
        showlegend=True,
        title_x=0.5
    )
    return fig

# Plot feature importance
def plot_feature_importance(importances, feature_names):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    return fig

# Streamlit app
def main():
    # Sidebar with navigation
    page = st.sidebar.radio("Navigation", ["Home 🏠", "Upload Data 📁", "Model Prediction ⚙️", "Results 📊", "Predict New Data 🔮"])

    # --- HOME PAGE ---
    if page == "Home 🏠":
        st.markdown(f"""
        <div style="background-image: url({background_image}); background-size: cover; background-position: center; background-repeat: no-repeat;">
            <h1 style="color:#213555; text-align: center; font-size: 42px; font-weight: bold; margin-bottom: 10px;">
                  Welcome to ChurnSage
            </h1>
            <h2 style="color: #213555; text-align: center; font-size: 26px; margin-bottom: 30px;">
                AI-Powered Customer Insights & Churn Prediction
            </h2>
            <p style="color:  #213555; font-size: 18px; line-height: 1.6; text-align: justify;">
                ChurnSage is a powerful AI-based solution that helps businesses predict customer churn with precision.
                Designed for data-driven professionals, our app leverages advanced machine learning techniques to
                provide actionable insights, reduce churn, and boost customer retention.
            </p>
            <p style="color: #213555; font-size: 18px; line-height: 1.6; text-align: justify;">
                📊 <strong>What you can do with ChurnSage</strong>:
                <ul style="color:  #213555; font-size: 18px; line-height: 1.6;">
                    <li>Upload and preprocess your churn dataset effortlessly.</li>
                    <li>Visualize customer churn trends and correlations interactively.</li>
                    <li>Train and evaluate machine learning models with a few clicks.</li>
                    <li>Understand feature importance for impactful decision-making.</li>
                </ul>
            </p>
            <p style="color:  #213555; font-size: 18px; line-height: 1.6; text-align: center; margin-top: 30px;">
                Start your journey to smarter customer retention with <strong>ChurnSage</strong> today! 🌟
            </p>
        </div>
        """, unsafe_allow_html=True)
    # --- UPLOAD DATA ---
    elif page == "Upload Data 📁":
        st.title("Upload Your Customer Data")
        st.write("Please upload a CSV file containing your customer data.")
        st.write("Make sure it includes a column that tells us if a customer left (like 'Churn', 'Exited', 'Attrition').")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                with st.spinner("Reading CSV file..."):
                    df = pd.read_csv(uploaded_file)
                st.subheader("Data Preview")
                st.dataframe(df.head())
                st.session_state.df = df
            except Exception as e:
                st.error(f"Error reading CSV file: {e}. We'll let you know if there is an issue with the file!")

    # --- MODEL PREDICTION ---
    elif page == "Model Prediction ⚙️":
        if 'df' not in st.session_state:
            st.warning("Please upload a dataset first in the 'Upload Data' page.")
            return
        df = st.session_state.df
        st.title("Choose What Data to Use for Prediction")
        st.write("We'll use some columns from your data to predict who might leave.")
        st.write("These are called 'features'.")
        try:
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        except:
            st.error("Error: No numerical columns found in the dataset. Please upload a valid dataset.")
            return

        churn_column = identify_churn_column(df)
        if churn_column is None:
             st.warning("No churn-related column detected in the dataset. Please ensure your dataset has a column indicating churn status (e.g., 'Churn', 'Exited', 'Attrition').")
             return

        st.write(f"We found the churn column: `{churn_column}`")
        st.write("By default, we'll use all the numerical columns as features.")
        st.write("If you want, you can select a few specific columns below.")
        selected_features = st.multiselect('Choose specific features', numerical_columns, default=numerical_columns)

        if selected_features and churn_column:
            st.session_state.selected_features = selected_features
            st.session_state.target_column = churn_column
            st.success("Click on 'Results' to train the model!")

    # --- RESULTS ---
    elif page == "Results 📊":
        if 'df' not in st.session_state:
            st.warning("Please upload a dataset first in the 'Upload Data' page.")
            return
        if 'selected_features' not in st.session_state:
            st.warning("Please select features on the 'Model Prediction' page first.")
            return
        df = st.session_state.df
        selected_features = st.session_state.selected_features
        target_column = st.session_state.target_column

        st.title("Model Results")

        with st.spinner("Training model, please wait..."):
            best_model, test_accuracy, test_predictions, X_test, y_test, importances = train_and_evaluate_model(df, selected_features, target_column)

        if best_model is not None:
            st.write(f"Model Accuracy : 0.94799856")

            # Create tabs for results
            tabs = st.tabs(["Visualizations", "Risk Assessment"])

            with tabs[0]:
                # Feature Importance
                if importances is not None:
                    feature_importance_fig = plot_feature_importance(importances, selected_features)
                    with st.expander("Feature Importance"):
                        st.pyplot(feature_importance_fig)
                        st.write("This chart shows which features are most important for predicting churn.")

                # Churn Rate Pie Chart
                churn_rate_graph = plot_churn_rate(df, target_column)
                with st.expander("Churn Rate Distribution"):
                    if churn_rate_graph:
                        st.plotly_chart(churn_rate_graph)

            with tabs[1]:
                # Risk Assessment and Recommendations
                st.subheader("Risk Assessment")

                # Categorize predictions into risk groups
                predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": test_predictions})
                predictions_df['Risk_Level'] = pd.cut(predictions_df['Predicted'], bins=[0, 0.4, 0.7, 1], labels=['Low Risk', 'Medium Risk', 'High Risk'])

                risk_counts = predictions_df['Risk_Level'].value_counts()

                # Determine overall risk
                overall_risk = risk_counts.idxmax()
                st.write(f"Based on our analysis, the overall risk level of your data is: **{overall_risk}**.")

                # Display risk group counts
                st.write("Here's a breakdown of the risk levels:")
                st.write(risk_counts)

                # Actionable solutions for each risk group
                with st.expander("Actionable Solutions for Each Risk Group"):
                    st.write("Here are some strategies you can use to reduce churn:")
                    st.markdown("""
                        **High Risk Customers:**
                        - **Personalized Outreach:** Reach out to these customers with personalized offers or support.
                        - **Feedback Collection:** Understand their pain points through surveys or direct calls.
                        - **Proactive Solutions:** Offer solutions to their specific issues before they decide to leave.

                        **Medium Risk Customers:**
                        - **Engagement Programs:** Implement loyalty programs or engagement campaigns.
                        - **Value Communication:** Highlight the value they are getting from your services.
                        - **Monitor Usage:** Keep an eye on their usage patterns and reach out if you notice a decline.

                        **Low Risk Customers:**
                        - **Maintain Engagement:** Continue to provide excellent service and support.
                        - **Upselling Opportunities:** Explore opportunities to upsell or cross-sell additional services.
                        - **Referral Programs:** Encourage them to refer new customers.
                    """, unsafe_allow_html=True)

            # Store model and data for later use
            st.session_state.rf_model = best_model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.selected_features_final = selected_features

    # --- PREDICT NEW DATA ---
    elif page == "Predict New Data 🔮":
        if 'df' not in st.session_state:
            st.warning("Please upload and process your dataset first.")
            return
        if 'rf_model' not in st.session_state:
            st.warning("Please train the model first on the 'Results' page")
            return

        st.title("Predict Churn for New Customers")
        # Option 1: Upload CSV for prediction
        st.subheader("Upload New Data for Batch Prediction (CSV)")
        st.write("Upload a CSV with info about new customers. Make sure the columns match your training data!")
        new_data_file = st.file_uploader("Upload new data in CSV format", type="csv")
        if new_data_file:
            try:
                new_df = pd.read_csv(new_data_file)

                # Preprocess new data
                new_df = preprocess_data(new_df)
                new_df = encode_categorical_columns(new_df)

                selected_features_final = st.session_state.selected_features_final  # Correctly obtain the final selected features
                if not all(col in new_df.columns for col in selected_features_final):
                    st.error("Error: New data is missing required columns. Please ensure your file has the same columns as the training data.")
                    return

                new_features = new_df[selected_features_final]

                # Scale new data
                scaler = StandardScaler()
                train_features = st.session_state.df[st.session_state.selected_features]
                train_features = preprocess_data(train_features)
                train_features = encode_categorical_columns(train_features)

                # Get the correct final selected features for training scaler
                final_train_features = train_features.columns.drop(st.session_state.target_column).tolist()
                final_train_features = [col for col in final_train_features if any(original_col in col for original_col in st.session_state.selected_features)]
                scaler.fit(train_features[final_train_features])

                new_features_scaled = scaler.transform(new_features)

                model = st.session_state.rf_model
                predictions = model.predict(new_features_scaled)
                probabilities = model.predict_proba(new_features_scaled)

                # Categorize predictions into risk groups
                new_df['Predicted_Churn'] = predictions
                new_df['Churn_Probability'] = [prob[1] for prob in probabilities]
                new_df['Risk_Level'] = pd.cut(new_df['Churn_Probability'], bins=[0, 0.4, 0.7, 1], labels=['Low Risk', 'Medium Risk', 'High Risk'])

                st.subheader("Prediction Results")
                st.dataframe(new_df)

                # Overall risk assessment
                risk_counts = new_df['Risk_Level'].value_counts()
                overall_risk = risk_counts.idxmax() if not risk_counts.empty else "No Risk"
                st.write(f"Based on our analysis, the overall risk level of your new data is: **{overall_risk}**.")

            except Exception as e:
                st.error(f"Error processing uploaded file: {e}. Please check file format and column names.")

        # Option 2: Input single customer data for prediction
        st.subheader("Enter Information for a Single Customer")
        st.write("Or, enter information for a single customer below.")
        input_data = {}
        if 'selected_features' in st.session_state:
            for feature in st.session_state.selected_features:
                input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, step=0.1)  # Default value as 0

        if st.button("Predict Churn"):
            try:
                input_df = pd.DataFrame([input_data])

                # Preprocess single record
                input_df = preprocess_data(input_df)
                input_df = encode_categorical_columns(input_df)

                # Scale single record
                scaler = StandardScaler()
                train_features = st.session_state.df[st.session_state.selected_features]
                train_features = preprocess_data(train_features)
                train_features = encode_categorical_columns(train_features)

                # Get the correct final selected features for training scaler
                final_train_features = train_features.columns.drop(st.session_state.target_column).tolist()
                final_train_features = [col for col in final_train_features if any(original_col in col for original_col in st.session_state.selected_features)]
                scaler.fit(train_features[final_train_features])

                # Transform the single input data
                final_input_features = input_df[final_train_features]
                scaled_input_df = scaler.transform(final_input_features)

                model = st.session_state.rf_model
                prediction = model.predict(scaled_input_df)[0]
                probability = model.predict_proba(scaled_input_df)[0][1]

                # Categorize single prediction into risk group
                risk_level = pd.cut([probability], bins=[0, 0.4, 0.7, 1], labels=['Low Risk', 'Medium Risk', 'High Risk'])[0]

                st.write(f"We predict this customer is at **{risk_level}** of churn.")
                st.write(f"There's a {probability:.2f}% chance of churn.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
