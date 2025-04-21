# AAPL Stock Price Prediction using Stacked LSTM

This project builds and evaluates a stacked LSTM neural network to predict Apple Inc. (AAPL) stock prices using historical time series data from 2018 to 2025.

---

## Project Highlights

- **Model**: 3-layer stacked LSTM neural network
- **Data**: 7 years of AAPL stock price data (2018–2025)
- **Preprocessing**: Applied `MinMaxScaler`, used sliding window (look_back=50) for sequential data
- **Performance**:
  - **Train RMSE**: 3.02
  - **Test RMSE**: 3.61

---

## Tech Stack

- Python
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

---

## Results Visualization

The model predicts future stock prices based on past 50 days’ data. Final predictions are inverse-transformed and visualized:

```python
# Ensure 'look_back' and 'lst_output_reshaped' are correctly defined
plt.plot(day_new, scaler.inverse_transform(df1[-50:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output_reshaped))
plt.show()

---


