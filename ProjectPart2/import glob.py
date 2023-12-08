import glob
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


path = 'C:\\Users\\ASUS\\Desktop\\423ProjectPart2\\data'
all_files = glob.glob(path + "/*_bist30.csv")

specified_files = [
    "20180101_20180401_bist30.csv", "20180402_20180701_bist30.csv",
    "20180702_20180930_bist30.csv", "20181001_20181230_bist30.csv",
    "20181231_20190331_bist30.csv", "20190401_20190630_bist30.csv",
    "20190701_20190929_bist30.csv", "20190930_20191229_bist30.csv",
    "20191230_20200329_bist30.csv", "20200330_20200628_bist30.csv",
    "20200629_20200927_bist30.csv", "20200928_20201227_bist30.csv"
]

files_2018_to_2021 = [file for file in all_files if any(spec_file in file for spec_file in specified_files)]

# Dosya yolları üzerinde döngü yaparak verileri birleştir
combined_data_AKBNK = pd.DataFrame()
combined_data_THYAO = pd.DataFrame()

for file_path in files_2018_to_2021:
    # Dosyayı oku ve 'timestamp' sütununu tarih-saat nesnesine dönüştürerek indeks yap
    data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')

    # 'Symbol' sütunu 'AKBNK' olanları seç
    akbnk_data = data[data['short_name'] == 'AKBNK']
    THYAO_data = data[data['short_name'] == 'THYAO']

    # Verileri birleştir
    combined_data_AKBNK = pd.concat([combined_data_AKBNK, akbnk_data])
    combined_data_THYAO = pd.concat([combined_data_THYAO, THYAO_data])

# Original tests for AKBNK
result_original_AKBNK = adfuller(combined_data_AKBNK['price'])
test_statistic_original_AKBNK = result_original_AKBNK[0]
p_value_original_AKBNK = result_original_AKBNK[1]
#print(f'Test Statistic (AKBNK Original): {test_statistic_original_AKBNK}')
#print(f'p-value (AKBNK Original): {p_value_original_AKBNK}')

#if p_value_original_AKBNK <= 0.05:
    #print('The AKBNK original time series is likely stationary.')
#else:
    #print('The AKBNK original time series is likely non-stationary.')

# Original tests for THYAO
result_original_THYAO = adfuller(combined_data_THYAO['price'])
test_statistic_original_THYAO = result_original_THYAO[0]
p_value_original_THYAO = result_original_THYAO[1]
#print(f'Test Statistic (THYAO Original): {test_statistic_original_THYAO}')
#print(f'p-value (THYAO Original): {p_value_original_THYAO}')

#if p_value_original_THYAO <= 0.05:
    #print('The THYAO original time series is likely stationary.')
#else:
    #print('The THYAO original time series is likely non-stationary.')

# Difference the data and store it in differenced_data for both AKBNK and THYAO
differenced_data_AKBNK = combined_data_AKBNK['price'].diff().dropna()
differenced_data_THYAO = combined_data_THYAO['price'].diff().dropna()

# Differenced tests for AKBNK
result_differenced_AKBNK = adfuller(differenced_data_AKBNK)
test_statistic_differenced_AKBNK = result_differenced_AKBNK[0]
p_value_differenced_AKBNK = result_differenced_AKBNK[1]
#print(f'Test Statistic (AKBNK Differenced): {test_statistic_differenced_AKBNK}')
#print(f'p-value (AKBNK Differenced): {p_value_differenced_AKBNK}')

#if p_value_differenced_AKBNK <= 0.05:
    #print('The AKBNK differenced time series is likely stationary.')
#else:
    #print('The AKBNK differenced time series is likely non-stationary.')

# Differenced tests for THYAO
result_differenced_THYAO = adfuller(differenced_data_THYAO)
test_statistic_differenced_THYAO = result_differenced_THYAO[0]
p_value_differenced_THYAO = result_differenced_THYAO[1]
#print(f'Test Statistic (THYAO Differenced): {test_statistic_differenced_THYAO}')
#print(f'p-value (THYAO Differenced): {p_value_differenced_THYAO}')

#if p_value_differenced_THYAO <= 0.05:
    #print('The THYAO differenced time series is likely stationary.')
#else:
    #print('The THYAO differenced time series is likely non-stationary.')

# Plotting the differenced data for AKBNK
plt.plot(differenced_data_AKBNK.index, differenced_data_AKBNK)
plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('Differenced AKBNK Price Over Time')
#plt.show()

# Plotting the differenced data for THYAO
plt.plot(differenced_data_THYAO.index, differenced_data_THYAO)
plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('Differenced THYAO Price Over Time')
#plt.show()

order_AKBNK = (1, 1, 1)  # Example order, you may adjust this

# Fit the ARIMA model
model_AKBNK = ARIMA(differenced_data_AKBNK, order=order_AKBNK)
model_AKBNK_fit = model_AKBNK.fit()

# Print the fitted model summary
print(model_AKBNK_fit.summary())

# Make predictions
steps = 10  # Change the number of steps as needed
forecast_AKBNK = model_AKBNK_fit.get_forecast(steps=steps)
forecast_index_AKBNK = pd.date_range(start=differenced_data_AKBNK.index[-1], periods=steps + 1, freq='B')[1:]

# Manually specify the ARIMA order (p, d, q)
order = (1, 1, 1)  # Example order, you may adjust this

# Fit the ARIMA model
model_THYAO = ARIMA(differenced_data_THYAO, order=order)
model_THYAO_fit = model_THYAO.fit()

# Print the fitted model summary
print(model_THYAO_fit.summary())

# Make predictions
steps = 10  # Change the number of steps as needed
forecast = model_THYAO_fit.get_forecast(steps=steps)
forecast_index = pd.date_range(start=differenced_data_THYAO.index[-1], periods=steps + 1, freq='B')[1:]

# Plotting the differenced AKBNK data
plt.plot(differenced_data_AKBNK.index, differenced_data_AKBNK, label='Differenced Data')

# Plotting the forecast
plt.plot(forecast_index_AKBNK, forecast_AKBNK.predicted_mean, label='Forecast', linestyle='dashed')

# Plotting confidence intervals
conf_int_AKBNK = forecast_AKBNK.conf_int()
plt.fill_between(forecast_index_AKBNK, conf_int_AKBNK['lower price'], conf_int_AKBNK['upper price'], alpha=0.2)

plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('AKBNK Differenced Price Forecast (Manual ARIMA)')
plt.legend()
plt.show()

# Plotting the differenced THYAO data
plt.plot(differenced_data_THYAO.index, differenced_data_THYAO, label='Differenced Data')

# Plotting the forecast
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', linestyle='dashed')

# Plotting confidence intervals
conf_int = forecast.conf_int()
plt.fill_between(forecast_index, conf_int['lower price'], conf_int['upper price'], alpha=0.2)

plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('THYAO Differenced Price Forecast (Manual ARIMA)')
plt.legend()
plt.show()