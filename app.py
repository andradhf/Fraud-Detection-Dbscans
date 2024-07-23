from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load and prepare the data
card = pd.read_csv("CreditCardTransaction.csv", delimiter=";")
card['TranxDate'] = pd.to_datetime(card['TranxDate'])
card_x = card.drop(["Department", "Division", "Year", "Month", "Merchant", "TranxDescription", "TranxDate"], axis=1)
card['TranxDate'] = (card['TranxDate'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
card_x['TranxDate'] = card['TranxDate']

# Sample and normalize data
card_x_sample = card_x.sample(n=50000, random_state=42)
x_array = np.array(card_x_sample)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)

# Set initial DBSCAN parameters
eps = 0.1
min_samples = 10
dbscans = DBSCAN(eps=eps, min_samples=min_samples)
dbscans.fit(x_scaled)
labels = dbscans.labels_
card_x_sample["kluster"] = labels

@app.route('/')
def index():
    # Display the home page
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    user_input = request.form.get('show_visualization')
    n_outliers = list(labels).count(-1)
    outlier_data = card_x_sample[card_x_sample['kluster'] == -1]

    if user_input == 'yes':
        fig, ax = plt.subplots()
        scatter = ax.scatter(x_scaled[:, 0], x_scaled[:, 1], c=card_x_sample['kluster'], marker="o", alpha=1)
        plt.title("Hasil Klustering DBSCAN")
        plt.colorbar(scatter)
        plt.legend()
        
        # Save plot to a BytesIO object and encode as base64
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('results.html', outlier_count=n_outliers, plot_url=plot_url, outliers=outlier_data.to_html())

    return render_template('results.html', outlier_count=n_outliers, plot_url=None, outliers=outlier_data.to_html())

if __name__ == '__main__':
    app.run(debug=True)
