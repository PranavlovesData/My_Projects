# app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scalers
with open('gradient_boosting_model.sav', 'rb') as file:
    model = pickle.load(file)

with open('scaler_X.sav', 'rb') as file:
    scaler_X = pickle.load(file)

with open('scaler_y.sav', 'rb') as file:
    scaler_y = pickle.load(file)

# Title of the app
st.title('Interactive Energy Production Prediction')

# Description and instructions
st.write("Adjust the sliders to modify input features and see real-time predictions for energy production.")

# Input features using sliders
st.header('Input Features')
# Ensure the default value (value) is within the range of min_value and max_value
feature1 = st.slider('Distance to Solar Noon', min_value=0, max_value=100, value=0, step=10)
feature2 = st.slider('Temperature', min_value=40, max_value=78, value=40, step=1)  # Default value set within range
feature3 = st.slider('Wind Direction', min_value=1, max_value=36, value=1, step=1)  # Default value set within range
feature4 = st.slider('Wind Speed', min_value=1, max_value=30, value=1, step=5)  # Default value set within range
feature5 = st.slider('Sky Cover', min_value=1, max_value=4, value=1, step=1)  # Default value set within range
feature6 = st.slider('Visibility', min_value=0, max_value=10, value=0, step=2)  # Default value set within range
feature7 = st.slider('Humidity', min_value=14, max_value=100, value=14, step=25)  # Default value set within range
feature8 = st.slider('Average Wind Speed', min_value=0, max_value=40, value=0, step=5)  # Default value set within range
feature9 = st.slider('Average Pressure', min_value=20, max_value=40, value=20, step=5)  # Default value set within range

# Store the inputs into a DataFrame
input_features = pd.DataFrame({
    'distance_to_solar_noon': [feature1],
    'temperature': [feature2],
    'wind_direction': [feature3],
    'wind_speed': [feature4],
    'sky_cover': [feature5],
    'visibility': [feature6],
    'humidity': [feature7],
    'average_wind_speed': [feature8],
    'average_pressure': [feature9],
    # Add more features as required
})

# Scale the input features
input_features_scaled = scaler_X.transform(input_features)

# Make predictions
prediction_scaled = model.predict(input_features_scaled)
prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

# Display the prediction
st.subheader('Predicted Energy Production')
st.write(f"**{prediction[0]:.2f} units**")


# Visualization: Bar Plot
st.subheader('Prediction Overview - Bar Plot')
fig, ax = plt.subplots()
ax.barh(['Energy Production'], [prediction[0]], color='green')
ax.set_xlim(0, max(100, prediction[0] * 1.1))
fig.tight_layout()  # Ensure tight layout
st.pyplot(fig)

# Visualization: Line Plot of Features vs. Prediction
st.subheader('Line Plot: Features vs. Predicted Energy Production')
features = list(input_features.columns)
feature_values = input_features.values.flatten()
fig, ax = plt.subplots()
ax.plot(features, feature_values, marker='o', linestyle='-', color='blue')
ax.set_title('Features vs. Predicted Energy Production')
ax.set_xlabel('Features')
ax.set_ylabel('Feature Values')
fig.tight_layout()  # Ensure tight layout
st.pyplot(fig)

# Visualization: Scatter Plot
st.subheader('Scatter Plot: Feature 1 vs. Predicted Energy Production')
fig, ax = plt.subplots()
ax.scatter(feature1, prediction[0], color='red', s=100)
ax.set_title('Feature 1 vs. Predicted Energy Production')
ax.set_xlabel('Feature 1 Value')
ax.set_ylabel('Predicted Energy Production')
fig.tight_layout()  # Ensure tight layout
st.pyplot(fig)

# Visualization: Pie Chart of Feature Contributions
st.subheader('Pie Chart: Feature Contributions')
fig, ax = plt.subplots()
ax.pie(feature_values, labels=features, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('husl', len(features)))
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
fig.tight_layout()  # Ensure tight layout
st.pyplot(fig)

# Add a custom footer or instructions
st.write("---")
st.write("**Instructions:** Use the sliders above to adjust the features and predict the energy production. The visualizations above provide a detailed view of the input features and the predicted output.")