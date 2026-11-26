Case Description

Problem. Stock indices are driven by the history of prior prices, which makes them a textbook use case for sequence models. Accurate price forecasts would be valuable for analysis and risk management, but markets are also noisy and close to efficient, so it is a good case for evaluating a model honestly.

Task. Given the previous 60 trading days, predict the next trading day's close of the NASDAQ Composite (IXIC).

Key design decisions

Decision	Reason
Predict daily log-returns, then convert back to price (price_t = close_{t-1} · exp(return_t))	IXIC rose from ~55 to ~14,000. A network fed raw scaled prices cannot output values outside its training range, so it fails on the later test years. Returns are stationary and on a stable scale.
Chronological 70 / 15 / 15 train / validation / test split	Random shuffling would leak future information into training.
Scaler fitted on training data only	Prevents data leakage.
Naive baseline ("tomorrow = today")	Prices change only ~1 % a day, so any model looks good on a price chart. The baseline shows whether the network adds value.
LSTM and SimpleRNN	Isolates the effect of LSTM gating.

Evaluation. One-step-ahead predictions on the unseen test period, measured by RMSE, MAE and MAPE on price, plus directional accuracy (was the up/down call right?).

Data Description

Source file: Market.csv 112,457 rows × 8 columns, Yahoo-Finance-style daily data.

Column	Meaning
Index	Index ticker
Date	Trading day (M/D/YYYY)
Open, High, Low, Close	Daily prices
Adj Close	Adjusted close
Volume	Traded volume

Coverage (14 indices)

Ticker	Index	From	To	Rows
000001.SS	Shanghai Composite	1997-07-02	2021-05-31	5,963
399001.SZ	Shenzhen Component	1997-08-22	2021-06-02	5,928
GDAXI	DAX (Germany)	1987-12-30	2021-05-31	8,606
GSPTSE	S&P/TSX Composite (Canada)	1979-06-29	2021-05-31	10,776
HSI	Hang Seng (Hong Kong)	1986-12-31	2021-05-31	8,750
IXIC	NASDAQ Composite (USA)	1971-02-05	2021-05-28	12,690
J203.JO	FTSE/JSE All Share (South Africa)	2012-02-08	2021-05-31	2,387
KS11	KOSPI (South Korea)	1996-12-11	2021-06-01	6,181
N100	Euronext 100	1999-12-31	2021-06-02	5,507
N225	Nikkei 225 (Japan)	1965-01-05	2021-06-03	14,500
NSEI	Nifty 50 (India)	2007-09-17	2021-05-31	3,381
NYA	NYSE Composite (USA)	1965-12-31	2021-05-28	13,948
SSMI	Swiss Market Index	1990-11-09	2021-05-31	7,830
TWII	Taiwan Weighted	1997-07-02	2021-05-31	6,010

Index used: IXIC. It has the longest gap-free history (the other long series have 4–626 missing rows).

Property	Value
Trading days	12,690 (1971-02-05 → 2021-05-28)
Missing values	0
Close price	min 54.87, max 14,138.78
Daily log-return	std 1.25 %, min −13.15 %, max +13.25 %
Volume	0 on 3,457 early rows → not used as a feature

Preprocessing

Features per day: ret (log-return of Close) and range = (High − Low) / Close, standardised using training statistics only.
Each sample is a window of 60 days × 2 features; the target is the next day's (standardised) log-return.
Split (chronological):
Set	Samples	Period
Train	8,840	1971-05-05 → 2006-05-10
Validation	1,894	2006-05-11 → 2013-11-15
Test	1,895	2013-11-18 → 2021-05-28

EDA

Performance Analysis

Training setup: Adam (lr 1e-3), MSE loss, batch size 64, up to 40 epochs, early stopping on validation loss (patience 8, best weights restored), learning-rate reduction on plateau.

Metrics

RMSE / MAE (index points) and MAPE (%) of the reconstructed price on the test set.
Direction accuracy — share of days where the sign of the predicted return matched the actual one.
Results on the test set (2013-11-18 → 2021-05-28)

Fill the LSTM and SimpleRNN rows with the table printed by the notebook's Evaluate cell (it prints a ready-to-paste Markdown table and saves images/results.csv). The baseline row is already computed from this exact test split.

Model	RMSE	MAE	MAPE %	Direction acc %
LSTM	run notebook	run notebook	run notebook	run notebook
SimpleRNN	run notebook	run notebook	run notebook	run notebook
Naive (yesterday's close)	99.935	60.475	0.832	56.464 (always-up)
Training curves

Loss curves

Predictions vs actual (last ~250 test days)

Predictions

How to read the results
Price error vs baseline. The predicted price line will track the actual line closely — but so does the naive line, because a day's move is only about 1 %. The model adds value only if its RMSE/MAE is meaningfully lower than the baseline row. Being roughly equal is the expected, honest outcome for daily stock data and is not a bug.
Direction accuracy. Markets rose over the test period, so "always up" alone scores 56.5 %. A model needs to beat that figure to show real directional skill; networks typically land around 50–55 %.
LSTM vs SimpleRNN. Compare their test metrics and loss curves. LSTM gates are designed to retain information across the 60-step window; a plain RNN is more prone to vanishing gradients and noisier training.
Overfitting check. A validation loss that rises while training loss keeps falling means the network is memorising noise; early stopping limits this.
Limitations & future work
Daily returns are mostly noise (efficient-market hypothesis) — predictability is inherently low.
One-step-ahead predictions are rebuilt from the actual previous close; a multi-day recursive forecast would drift much more.
Only two features were used. Ideas: volume (for indices with valid data), moving averages, RSI, other indices as inputs, GRU/Bidirectional layers, tuning WINDOW, and walk-forward validation.
