# Stock Market Prediction System
This repository provides an AI-powered Stock Market Prediction System that leverages a Time Embedded Transformer Model to forecast next-day stock prices for a specified company. The system integrates historical stock data and recent news to improve prediction accuracy.

---

# Features
- Historical Data Collection: Fetches stock price data for the selected company.
- News Integration: Collects related news to enhance the prediction process.
- AI-Powered Predictions: Utilizes a transformer-based model to forecast stock prices.
- Data Visualization: Plots actual vs. predicted prices for training and validation datasets.
- Streamlit Web Interface: User-friendly UI for interaction with the system.

---

# Repository Structure
```
.
├── main.py               # Main script for backend stock prediction.
├── app.py                # Frontend interface powered by Streamlit.
├── helper/               # Contains utility modules:
│   ├── stockCollection.py       # Fetches historical stock prices.
│   ├── newsCollection.py        # Collects company-related news.
│   ├── dataprocess.py           # Pre-processes data for the model.
│   └── timeEmbeddingTransformer.py  # Implements the transformer model.
├── companies/            # Auto-generated directory for company data and models.
└── README.md             # Project documentation.
```

---

# Requirements
```
Python 3.8+
Libraries:
streamlit
numpy
pandas
matplotlib
tensorflow
scikit-learn
```


# How to Use
## Clone the repository:
```
git clone https://github.com/your-username/stock-market-prediction.git
cd stock-market-prediction
```

## Install dependencies:
```
pip install -r requirements.txt
```

## Run the Streamlit app:
```
streamlit run app.py
```

## Enter a company's name in the text input field and click "Predict Next Day's Stock Value" to get the forecast.

---

# Key Functionality
## main.py
### StockPrediction(comp):
- Collects and updates stock data and news for the specified company.
- Trains the transformer model if no prior trained model exists.
- Predicts the next day's stock price based on trends.

## app.py
- Provides an interactive interface for users to input company names and view predictions.

---

# Outputs
- Predicted Stock Prices: Displays today's stock price and the predicted price for tomorrow.
- Performance Metrics:
  - Training and validation loss.
  - Training and validation MSE.
- Visualization: Generates plots showing predicted vs. actual stock prices.

---

# Future Enhancements
- Add support for multi-stock prediction.
- Incorporate sentiment analysis for news data.
- Optimize the transformer model for faster training.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
