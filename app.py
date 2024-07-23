from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def run_dbscan():
    # Load data
    card = pd.read_csv("CreditCardTransaction.csv", delimiter=";")

    # Convert TranxDate column to datetime
    card['TranxDate'] = pd.to_datetime(card['TranxDate'])

    # Select attributes (excluding non-numeric columns)
    card_x = card.drop(["Department", "Division", "Year", "Month", "Merchant", "TranxDescription", "TranxDate"], axis=1)

    # Convert TranxDate to numeric (number of days since epoch)
    card['TranxDate'] = (card['TranxDate'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    card_x['TranxDate'] = card['TranxDate']

    # Use a subset of the data (e.g., 50,000 rows for testing)
    card_x_sample = card_x.sample(n=50000, random_state=42)
    x_array = np.array(card_x_sample)

    # Normalization process
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_array)

    # Set initial DBSCAN parameters
    eps = 0.1  # Start with a smaller value of eps
    min_samples = 10  # Start with a larger value of min_samples

    # Insert array to DBSCAN
    dbscans = DBSCAN(eps=eps, min_samples=min_samples)
    dbscans.fit(x_scaled)

    # Menampilkan jumlah cluster
    labels = dbscans.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)

    # Show outlier data
    outlier_data = []
    for i in range(len(card_x_sample)):
        if dbscans.labels_[i] == -1:
            # Convert numeric date back to datetime
            transaction_date = pd.to_datetime(card.values[i, 6], unit='s')
            outlier_data.append({
                "Transaction Date": transaction_date,
                "Merchant": card.values[i, 4],
                "Amount": card.values[i, 7]
            })

    return n_clusters_, n_outliers, outlier_data, x_scaled, dbscans.labels_

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    show_visualization = request.form['show_visualization']
    
    n_clusters_, n_outliers, outlier_data, x_scaled, labels = run_dbscan()
    
    if show_visualization == 'yes':
        fig, ax = plt.subplots()
        scatter = ax.scatter(x_scaled[:, 0], x_scaled[:, 1], c=labels, marker="o", alpha=1)
        plt.title("Hasil Klustering DBSCAN")
        plt.colorbar(scatter)
        plt.legend()

        # Save plot to a BytesIO object and encode as base64
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('results.html', outlier_count=n_outliers, plot_url=plot_url, outliers=outlier_data)

    return render_template('results.html', outlier_count=n_outliers, plot_url=None, outliers=outlier_data)

if __name__ == '__main__':
    app.run(debug=True)
