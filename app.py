import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title of the dashboard
st.title("Crime Analysis Dashboard with Predictions")

# Load the dataset
df = pd.read_csv('/workspaces/crime_safety_dataset.csv')

# Clean data (remove invalid rows)
df = df.dropna()

# Encode categorical variables for machine learning
le_crime = LabelEncoder()
le_city = LabelEncoder()
df['crime_type_encoded'] = le_crime.fit_transform(df['crime_type'])
df['city_encoded'] = le_city.fit_transform(df['city'])

# Features and target (predict Robbery probability)
X = df[['victim_age', 'city_encoded', 'crime_type_encoded']]
y = (df['crime_type'] == 'Robbery').astype(int)

# Train a simple logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
df['robbery_probability'] = model.predict_proba(X)[:, 1]

# Display bar chart of crime types
st.subheader("Crime Type Distribution")
st.bar_chart(df.groupby('crime_type')['id'].count())

# Display top 5 cities by robbery probability
st.subheader("Top 5 Cities by Robbery Probability")
top_cities = df.groupby('city')['robbery_probability'].mean().sort_values(ascending=False).head(5)
st.bar_chart(top_cities)

# Export updated dataset with predictions
df.to_csv('/workspaces/crime_safety_dataset_updated.csv', index=False)
st.success("Updated dataset with predictions exported to crime_safety_dataset_updated.csv")

# Optional: Add a download button (for manual use)
st.download_button(
    label="Download Updated CSV",
    data=df.to_csv(index=False),
    file_name='crime_safety_dataset_updated.csv',
    mime='text/csv'
)
