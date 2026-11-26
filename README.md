# RNN & LSTM Practice — Stock Price Prediction

**Problem:** predict the next trading day's NASDAQ Composite (IXIC) close from the previous 60 days.
**Data:** `Market.csv` — daily OHLCV for 14 world stock indices, 1965–2021.
**Models:** 2-layer LSTM and 2-layer SimpleRNN (Keras), compared against a naive "tomorrow = today" baseline.

## Run in Google Colab
1. Go to https://colab.research.google.com -> File -> Upload notebook -> choose `LSTM_Stock_Prediction.ipynb`.
2. Upload `Market.csv` in the left Files panel (or just run the notebook: it will prompt for the upload).
3. Runtime -> Run all (CPU is fine, a few minutes).

Change `INDEX = "IXIC"` in the data cell to try another index.
