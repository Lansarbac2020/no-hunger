import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Load data
@st.cache_data
def load_data():
    # Update this path to where your data file is located
    df_merged = pd.read_csv('merged_data.csv')
    return df_merged

df_merged = load_data()

# Define your features and targets
features = ['ghi_2000', 'ghi_2007', 'ghi_2014', 'ghi_2022'] # Update with your actual feature names
targets = ['2022_undernourished', '2022_wasting', '2022_stunting', '2022_mortality']

def predict_for_country(country_name, future_year):
    country_data = df_merged[df_merged['country'] == country_name]
    
    if country_data.empty:
        st.write(f"No data available for country: {country_name}")
        return None
    
    predicted_values = {}
    models = {}
    for target in targets:
        model = joblib.load(f'{target}_model.pkl')
        models[target] = model

    missing_features = [col for col in features if col not in country_data.columns]
    if missing_features:
        st.write(f"Missing features in data: {missing_features}")
        return None
    
    for target in targets:
        model = models[target]
        X_country = country_data[features]
        predicted_values[target] = model.predict(X_country).mean()

    st.write(f'Predicted values for {country_name} in {future_year}:')
    for target in targets:
        new_target_name = target.replace("2022_", "") + f' in {future_year}'
        st.write(f'  {new_target_name}: {predicted_values[target]:.2f}')
    
    X_future = pd.DataFrame({
        '2022_undernourished': [predicted_values.get('2022_undernourished', 0)],
        '2022_wasting': [predicted_values.get('2022_wasting', 0)],
        '2022_stunting': [predicted_values.get('2022_stunting', 0)],
        '2022_mortality': [predicted_values.get('2022_mortality', 0)]
    })
    
    ghi_model = joblib.load('ghi_2030_model.pkl')
    try:
        GHI_future = ghi_model.predict(X_future)
        predicted_values['GHI'] = GHI_future[0]
    except ValueError as e:
        st.write(f"Error predicting GHI: {e}")
        return None

    st.write(f'  GHI in {future_year}: {predicted_values["GHI"]:.2f}')
    
    years = ['2000', '2007', '2014', '2022', str(future_year)]
    historical_data = {
        'Year': years,
        'Undernourished': list(country_data[['2000_undernourished', '2007_undernourished', '2014_undernourished', '2022_undernourished']].values.flatten()) + [predicted_values.get('2022_undernourished', 0)],
        'Wasting': list(country_data[['2000_wasting', '2007_wasting', '2014_wasting', '2022_wasting']].values.flatten()) + [predicted_values.get('2022_wasting', 0)],
        'Stunting': list(country_data[['2000_stunting', '2007_stunting', '2014_stunting', '2022_stunting']].values.flatten()) + [predicted_values.get('2022_stunting', 0)],
        'Mortality': list(country_data[['2000_mortality', '2007_mortality', '2014_mortality', '2022_mortality']].values.flatten()) + [predicted_values.get('2022_mortality', 0)]
    }
    
    df_historical = pd.DataFrame(historical_data)
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    sns.lineplot(data=df_historical, x='Year', y='Undernourished', ax=ax[0, 0], marker='o')
    ax[0, 0].set_title('Undernourished Over Time')
    
    sns.lineplot(data=df_historical, x='Year', y='Wasting', ax=ax[0, 1], marker='o')
    ax[0, 1].set_title('Wasting Over Time')
    
    sns.lineplot(data=df_historical, x='Year', y='Stunting', ax=ax[1, 0], marker='o')
    ax[1, 0].set_title('Stunting Over Time')
    
    sns.lineplot(data=df_historical, x='Year', y='Mortality', ax=ax[1, 1], marker='o')
    ax[1, 1].set_title('Mortality Over Time')
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Streamlit app
st.title('GHI Prediction and Visualization')

country_name = st.text_input('Enter Country Name:', 'Türkiye')
future_year = st.number_input('Enter Future Year:', min_value=2023, value=2030)

if st.button('Predict'):
    plot_buffer = predict_for_country(country_name, future_year)
    if plot_buffer:
        st.image(plot_buffer)
