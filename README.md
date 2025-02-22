ChurnSage: AI-Powered Customer Churn Prediction and Risk Assessment System

Project Overview 
ChurnSage is an advanced AI-driven platform designed to help businesses predict customer churn with high accuracy. It empowers organizations to proactively identify at-risk customers, understand key churn drivers, and implement strategic interventions to improve customer retention. The system is built using Streamlit for an interactive web-based UI, **Scikit-Learn** for machine learning, **Plotly** and Seaborn for visual analytics, and Pandas for efficient data handling.  

This solution is ideal for customer success teams, business analysts, and data-driven organizations that want to enhance customer retention strategies through AI-driven insights.

---

Key Features and Functionalities

 🏠 Home Page
- A visually appealing landing page that introduces users to ChurnSage and its capabilities.  
- Provides an overview of how the system can help businesses analyze customer behavior and predict churn.  

 📁 Upload Customer Data  
- Users can upload a CSV dataset containing customer information.  
- The system automatically detects and processes numerical and categorical data.  
- Missing values are handled using imputation techniques (median for numerical values, mode for categorical values).  
- The system attempts to identify the churn-related column automatically.  

 ⚙️ Model Training & Prediction
- Users can select relevant features to train a **Random Forest Classifier**.  
- The model is trained using **RandomizedSearchCV** to optimize hyperparameters.  
- The dataset is split into training and testing sets to evaluate model performance.  
- **Feature importance analysis** helps users understand which factors contribute most to churn.  
- The final model is stored for real-time predictions on new data.  

📊 Results & Visual Analytics  
- Displays model accuracy and key performance metrics.  
- Generates a **churn rate distribution pie chart** using Plotly to visualize the proportion of customers leaving.  
- Provides an interactive **feature importance bar chart** to highlight the most influential factors in customer churn.  
- Includes a **Risk Assessment System**, categorizing customers into **Low Risk, Medium Risk, and High Risk** groups based on churn probability.  
- Provides **strategic recommendations** tailored to different risk groups to reduce churn effectively.  

🔮 Predict Churn for New Customers 
- Users can upload a new dataset for **batch predictions** on unseen customers.  
- The system preprocesses the new data to match the trained model’s feature set.  
- Predictions include **churn probability and risk categorization**.  
- Alternatively, users can manually enter customer details for **single-customer predictions**.  
- Provides actionable insights based on prediction results.  

---

Technical Stack  
- **Frontend/UI**: Streamlit  
- **Machine Learning**: Scikit-Learn (Random Forest, One-Hot Encoding, Imputation, Standard Scaling)  
- **Data Processing**: Pandas, NumPy  
- **Data Visualization**: Plotly, Seaborn, Matplotlib  
- **Model Optimization**: RandomizedSearchCV for hyperparameter tuning  

---

How ChurnSage Works
1. **Upload Customer Data** → Preprocess and clean data.  
2. **Feature Selection & Model Training** → Identify key attributes, train model with hyperparameter tuning.  
3. **Evaluate Performance** → Visualize churn distribution, analyze feature importance.  
4. **Make Predictions** → Apply trained model to new customers, predict churn risk.  
5. **Provide Actionable Insights** → Categorize customers by risk level and suggest retention strategies.  

---

Potential Use Cases 
- **Telecom Industry**: Predict which subscribers may cancel their service.  
- **E-commerce & Retail**: Identify customers likely to stop shopping and implement targeted retention campaigns.  
- **Banking & Finance**: Detect customers at risk of account closure.  
- **Subscription-based Services**: Forecast cancellations and optimize renewal strategies.  

---

Conclusion 
ChurnSage is a powerful AI-driven solution for businesses seeking to understand customer behavior and reduce churn. By leveraging machine learning, data visualization, and risk assessment methodologies, this system provides actionable insights that drive customer retention and business growth. 🚀
