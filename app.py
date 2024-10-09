import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from collections import OrderedDict

# Set up the company dictionary
companies_dict = OrderedDict({
    'Amazon': 'AMZN',
    'Apple': 'AAPL',
    'Walgreen': 'WBA',
    'Northrop Grumman': 'NOC',
    'Boeing': 'BA',
    'Lockheed Martin': 'LMT',
    'McDonalds': 'MCD',
    'Intel': 'INTC',
    'Navistar': 'NAV',
    'IBM': 'IBM',
    'Texas Instruments': 'TXN',
    'MasterCard': 'MA',
    'Microsoft': 'MSFT',
    'General Electrics': 'GE',
    'Symantec': 'SYMC',
    'American Express': 'AXP',
    'Pepsi': 'PEP',
    'Coca Cola': 'KO',
    'Johnson & Johnson': 'JNJ',
    'Toyota': 'TM',
    'Honda': 'HMC',
    'Mitsubishi': 'MSBHY',
    'Sony': 'SNE',
    'Exxon': 'XOM',
    'Chevron': 'CVX',
    'Valero Energy': 'VLO',
    'Ford': 'F',
    'Bank of America': 'BAC'
})

# Streamlit UI
st.title("Stock Market Clustering with K-Means")
st.write("Select companies for analysis:")

# Select companies
selected_companies = st.multiselect("Choose companies:", options=list(companies_dict.keys()))

# Set date range for fetching stock data
start_date = st.date_input("Start date:", value=pd.to_datetime('2015-01-01'))
end_date = st.date_input("End date:", value=pd.to_datetime('2017-12-31'))

if st.button("Fetch Data"):
    # Fetch stock data using yfinance
    tickers = [companies_dict[company] for company in selected_companies]
    df = yf.download(tickers, start=start_date, end=end_date)

    # Display data summary
    st.subheader("Data Summary")
    if not df.empty:
        for company in selected_companies:
            ticker = companies_dict[company]
            st.write(f"{company} Closing Prices Summary:")
            # Accessing closing prices correctly for multi-index DataFrame
            if ticker in df['Close']:
                st.dataframe(df['Close'][ticker].describe())
            else:
                st.warning(f"No data available for {company} in the specified date range.")
    else:
        st.warning("No stock data fetched. Please check your selections and try again.")

    # Stock movements
    stock_open = np.array(df['Open']).T
    stock_close = np.array(df['Close']).T
    movements = stock_close - stock_open

    # Clean up NaN values
    movements_cleaned = movements[~np.isnan(movements).any(axis=1)]

    # Ensure there are enough samples for clustering
    num_samples = movements_cleaned.shape[0]
    max_clusters = min(num_samples, 3)  # Set max clusters to number of samples or 3

    if num_samples < 3:
        st.error("Not enough data points for clustering. Please select more companies or a different date range.")
    else:
        # Normalize movements
        norm_movements = Normalizer().fit_transform(movements_cleaned)

        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(norm_movements)

        # Clustering with KMeans
        kmeans = KMeans(n_clusters=max_clusters, max_iter=1000)
        pipeline = make_pipeline(Normalizer(), kmeans)
        pipeline.fit(movements_cleaned)
        labels = pipeline.predict(movements_cleaned)

        # Map labels to companies
        companies_cleaned = np.array(selected_companies)[~np.isnan(movements).any(axis=1)]
        df1 = pd.DataFrame({'labels': labels, 'companies': companies_cleaned}).sort_values(by=['labels'], axis=0)

        # Cluster Insights
        st.subheader("Cluster Insights")
        for label in df1['labels'].unique():
            st.write(f"\n**Cluster {label}:**")
            st.write(df1[df1['labels'] == label]['companies'].values)

        # Visualization
        st.subheader("Movement Visualization")
        fig = go.Figure(data=[go.Scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], mode='markers',
                                           marker=dict(color=labels, showscale=True))])
        fig.update_layout(title="K-Means Clustering of Stock Movements",
                          xaxis_title="Principal Component 1",
                          yaxis_title="Principal Component 2")
        st.plotly_chart(fig)

        # Additional plots for selected companies
        for company in selected_companies:
            st.subheader(f"{company} Stock Movement")
            plt.figure(figsize=(10, 5))
            plt.plot(df['Open'][companies_dict[company]], label='Opening Price')
            plt.plot(df['Close'][companies_dict[company]], label='Closing Price')
            plt.title(f"{company} Stock Prices")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(plt)

        # Candlestick chart for the first selected company
        if selected_companies:
            first_company = companies_dict[selected_companies[0]]
            fig_candlestick = go.Figure(data=[go.Candlestick(x=df.index,
                                                              open=df['Open'][first_company],
                                                              high=df['High'][first_company],
                                                              low=df['Low'][first_company],
                                                              close=df['Close'][first_company])])
            fig_candlestick.update_layout(title=f"{selected_companies[0]} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_candlestick)

        # Display normalized movements
        st.subheader("Normalized Movements")
        plt.figure(figsize=(10, 5))
        plt.title("Normalized Movements of Selected Companies")
        for i in range(len(companies_cleaned)):
            plt.plot(norm_movements[i], label=companies_cleaned[i])
        plt.xlabel("Days")
        plt.ylabel("Normalized Movement")
        plt.legend()
        st.pyplot(plt)
