from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

# Load data
def load_data():
    # Update this path to where your data file is located
    df_merged = pd.read_csv('merged_data.csv')
    return df_merged

df_merged = load_data()

# Define your features and targets
features = ['ghi_2000', 'ghi_2007', 'ghi_2014', 'ghi_2022']  # Update with your actual feature names
targets = ['2022_undernourished', '2022_wasting', '2022_stunting', '2022_mortality']

def predict_for_country(country_name, future_year):
    country_data = df_merged[df_merged['country'] == country_name]
    
    if country_data.empty:
        return None, "No data available for country: {}".format(country_name), None
    
    predicted_values = {}
    models = {}
    for target in targets:
        model = joblib.load(f'{target}_model.pkl')
        models[target] = model

    missing_features = [col for col in features if col not in country_data.columns]
    if missing_features:
        return None, "Missing features in data: {}".format(missing_features), None
    
    for target in targets:
        model = models[target]
        X_country = country_data[features]
        predicted_values[target] = model.predict(X_country).mean()

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
        return None, "Error predicting GHI: {}".format(e), None

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
    
    return {
        'Undernourished in 2030': f"{predicted_values.get('2022_undernourished', 0):.2f}",
        'Wasting in 2030': f"{predicted_values.get('2022_wasting', 0):.2f}",
        'Stunting in 2030': f"{predicted_values.get('2022_stunting', 0):.2f}",
        'Mortality in 2030': f"{predicted_values.get('2022_mortality', 0):.2f}",
        'GHI in 2030': f"{predicted_values.get('GHI', 0):.2f}"
    }, None, buf

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        country_name = request.form['country_name']
        future_year = int(request.form['future_year'])
        
        predictions, error_message, plot_buffer = predict_for_country(country_name, future_year)
        
        if error_message:
            return render_template('index.html', error=error_message)
        
        plot_url = None
        if plot_buffer:
            plot_url = 'data:image/png;base64,' + base64.b64encode(plot_buffer.read()).decode('utf-8')
        
        return render_template('index.html', predictions=predictions, plot_url=plot_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    # app.run(debug=True)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)