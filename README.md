# Real-Time Stock Movement Prediction Using LSTM and FinBERT Sentiment Analysis

## Overview

This project is an integrated version of [stock-predictor-using-randomforest](https://github.com/m-turnergane/stock-predictor-using-randomforest) by Muhammad Gane. The original project utilized Random Forest for stock prediction, but lacked advanced sentiment integration and real-time forecasting.

- This project expanded the functionality by integrating Bursa Malaysia stock market data for localized insights.
- Incorporating FinBERT-based sentiment analysis to assess market sentiment from financial news and reports.
- Implementing an LSTM model for time-series forecasting of stock movements.
- Developing a real-time predictive dashboard for dynamic visualization and decision support.

---

## Objective

- Integrate Bursa Malaysia market data with sentiment analysis
- Apply FinBERT for financial sentiment extraction
- Use LSTM to predict short-term stock movement
- Build real-time dashboard for traders.

---

## Installation
```bash
# Clone the repository
git clone https://github.com/weiiilunnnn/financial-sentiment-predictor.git
cd financial-sentiment-predictor

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

  ## Next Step

  - Data collection, preprocessing, sentiment analysis with FinBERT, LSTM model training and evaluation and Dashboard inergration.

  - > **Status**: Work in Progress — Initial setup and data sourcing stage.

---

## Tech Stack

- Data Source: Yahoo Finance, NewsAPI, X (formerly Twitter)
- Sentiment: FinBERT
- Moddeling: Python (TensorFlow/PyTorch)
- Visualization: Plotly, Streamlit

---

## 📬 Contact

Build by [@weiiilunnnn](https://github.com/weiiilunnnn)

For questions, reach out via GitHub Issues or fork this project.
