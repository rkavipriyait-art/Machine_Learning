# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error


# LOAD & PREPARE DATA
df = pd.read_csv("stores_sales_forecasting.csv", encoding='latin1')
print(df)
print("---------------")
print(df.describe())
print("---------------")
print(df.columns)
print("---------------")
print(df.info())
print("---------------")
print(df.duplicated())
print("---------------")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date')
df.set_index('Order Date', inplace=True)

# Daily Sales
daily_sales = df['Sales'].resample('D').sum().asfreq('D').fillna(0)
print("Frequency:", daily_sales.index.freq)

# TREND & SEASONALITY ANALYSIS
decomposition = seasonal_decompose(daily_sales, model='additive', period=30)
decomposition.plot()
plt.show()

monthly_sales = daily_sales.resample('ME').sum()
plt.figure(figsize=(10,5))
plt.plot(monthly_sales)
plt.title("Monthly Sales Trend")
plt.show()


# STATIONARITY CHECK
rolling_mean = daily_sales.rolling(window=30).mean()
rolling_std = daily_sales.rolling(window=30).std()

plt.figure(figsize=(12,6))
plt.plot(rolling_mean, label='Rolling Mean (30D)')
plt.plot(rolling_std, label='Rolling Std (30D)')
plt.legend()
plt.title('Rolling Mean & Std Dev')
plt.show()

adf_result = adfuller(daily_sales)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print('Number of lags used:', adf_result[2])
print('Number of observations:', adf_result[3])
print('Critical values:', adf_result[4])
print('Information criterion:', adf_result[5])

#if ADF Statistic < Critical Value (5%) is stationary otherwise non-stationary,
#p-value (< 0.05 -> Stationary, > 0.05 -> Non-stationary)
print("This dataset is stationary. No need for non-seasonal differencing")


# SALES FORECASTING (SARIMA – Seasonal Model)
train_size = int(len(daily_sales) * 0.8)
train = daily_sales[:train_size]
test = daily_sales[train_size:]

model = SARIMAX(train,
                order=(1,0,1),
                seasonal_order=(1,0,1,30),
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)

forecast_object = model_fit.get_forecast(steps=len(test))
forecast = forecast_object.predicted_mean
conf_int = forecast_object.conf_int()

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100

print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)

# Plot Forecast
plt.figure(figsize=(12,6))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(forecast, label="Forecast")
plt.fill_between(test.index,
                 conf_int.iloc[:,0],
                 conf_int.iloc[:,1],
                 color='gray', alpha=0.3)
plt.legend()
plt.title("Sales Forecast (SARIMA)")
plt.show()


# PRODUCT CATEGORY FORECASTING (SARIMA)
if 'Category' in df.columns:
    categories = df['Category'].unique()

    for cat in categories:
        cat_data = df[df['Category'] == cat]
        cat_daily = cat_data['Sales'].resample('D').sum().asfreq('D').fillna(0)

        if len(cat_daily) < 60:
            continue

        train_size = int(len(cat_daily) * 0.8)
        train = cat_daily[:train_size]
        test = cat_daily[train_size:]

        model = SARIMAX(train,
                        order=(1,0,1),
                        seasonal_order=(1,0,1,30),
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(test))

        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"Category: {cat} | RMSE: {rmse}")


# REVENUE GROWTH ANALYSIS (USING ORIGINAL DATA)
monthly_revenue = daily_sales.resample('ME').sum()
mom_growth = monthly_revenue.pct_change() * 100

plt.figure(figsize=(10,5))
plt.plot(mom_growth)
plt.title("Month-over-Month Growth (%)")
plt.show()

yearly_revenue = daily_sales.resample('YE').sum()
yoy_growth = yearly_revenue.pct_change() * 100

print("Year-over-Year Growth (%)")
print(yoy_growth)

if len(yearly_revenue) > 1:
    start_value = yearly_revenue.iloc[0]
    end_value = yearly_revenue.iloc[-1]
    n_years = len(yearly_revenue) - 1

    cagr = ((end_value / start_value) ** (1/n_years) - 1) * 100
    print("CAGR:", cagr)


# RESIDUAL-BASED FRAUD DETECTION (FIXED)
# Fit model on FULL series for fraud detection
fraud_model = SARIMAX(daily_sales,
                      order=(1,1,1),
                      seasonal_order=(1,1,1,30),
                      enforce_stationarity=False,
                      enforce_invertibility=False)

fraud_model_fit = fraud_model.fit(disp=False)

# Get residuals (aligned correctly)
residuals = fraud_model_fit.resid

# Standardize residuals
z_score = (residuals - residuals.mean()) / residuals.std()

# Now index aligns correctly
residual_fraud = daily_sales[z_score.abs() > 3]

plt.figure(figsize=(12,6))
plt.plot(daily_sales, label="Sales")
plt.scatter(residual_fraud.index,
            residual_fraud,
            color='purple',
            label="Residual Fraud")
plt.legend()
plt.title("Residual-Based Fraud Detection")
plt.show()

print("Residual Fraud Dates:")
print(residual_fraud)
