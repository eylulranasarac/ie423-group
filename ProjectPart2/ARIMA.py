import glob
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

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
combined_data_GARAN = pd.DataFrame()

for file_path in files_2018_to_2021:
    # Dosyayı oku ve 'timestamp' sütununu tarih-saat nesnesine dönüştürerek indeks yap
    data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')

    # 'Symbol' sütunu 'AKBNK' olanları seç
    akbnk_data = data[data['short_name'] == 'AKBNK']
    garan_data = data[data['short_name'] == 'GARAN']

    # Verileri birleştir
    combined_data_AKBNK = pd.concat([combined_data_AKBNK, akbnk_data])
    combined_data_GARAN = pd.concat([combined_data_GARAN, garan_data])

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

# Original tests for GARAN
result_original_GARAN = adfuller(combined_data_GARAN['price'])
test_statistic_original_GARAN = result_original_GARAN[0]
p_value_original_GARAN = result_original_GARAN[1]
#print(f'Test Statistic (GARAN Original): {test_statistic_original_GARAN}')
#print(f'p-value (GARAN Original): {p_value_original_GARAN}')

#if p_value_original_GARAN <= 0.05:
    #print('The GARAN original time series is likely stationary.')
#else:
    #print('The GARAN original time series is likely non-stationary.')

# Difference the data and store it in differenced_data for both AKBNK and GARAN
differenced_data_AKBNK = combined_data_AKBNK['price'].diff().dropna()
differenced_data_GARAN = combined_data_GARAN['price'].diff().dropna()

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

# Differenced tests for GARAN
result_differenced_GARAN = adfuller(differenced_data_GARAN)
test_statistic_differenced_GARAN = result_differenced_GARAN[0]
p_value_differenced_GARAN = result_differenced_GARAN[1]
#print(f'Test Statistic (GARAN Differenced): {test_statistic_differenced_GARAN}')
#print(f'p-value (GARAN Differenced): {p_value_differenced_GARAN}')

#if p_value_differenced_GARAN <= 0.05:
    #print('The GARAN differenced time series is likely stationary.')
#else:
    #print('The GARAN differenced time series is likely non-stationary.')

# Plotting the differenced data for AKBNK
plt.plot(differenced_data_AKBNK.index, differenced_data_AKBNK)
plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('Differenced AKBNK Price Over Time')
#plt.show()

# Plotting the differenced data for GARAN
plt.plot(differenced_data_GARAN.index, differenced_data_GARAN)
plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('Differenced GARAN Price Over Time')
#plt.show()


model_AKBNK = auto_arima(differenced_data_AKBNK, seasonal=False, trace=True, suppress_warnings=True)
model_AKBNK.fit(differenced_data_AKBNK)

# Auto ARIMA model for GARAN
model_GARAN = auto_arima(differenced_data_GARAN, seasonal=False, trace=True, suppress_warnings=True)
model_GARAN.fit(differenced_data_GARAN)

# Print the model summary for AKBNK
print("Auto ARIMA Model Summary for AKBNK:")
print(model_AKBNK.summary())

# Print the model summary for GARAN
print("Auto ARIMA Model Summary for GARAN:")
print(model_GARAN.summary())
# Forecast future values for AKBNK
future_steps = 10  # You can adjust the number of steps into the future
forecast_AKBNK, conf_int_AKBNK = model_AKBNK.predict(n_periods=future_steps, return_conf_int=True)

# Forecast future values for GARAN
forecast_GARAN, conf_int_GARAN = model_GARAN.predict(n_periods=future_steps, return_conf_int=True)

# Plotting the results for AKBNK
plt.plot(differenced_data_AKBNK.index, differenced_data_AKBNK, label='Actual AKBNK Differenced Data')
plt.plot(pd.date_range(start=differenced_data_AKBNK.index[-1], periods=future_steps+1, freq='B')[1:], forecast_AKBNK, label='Forecasted AKBNK Differenced Data', color='red')
plt.fill_between(pd.date_range(start=differenced_data_AKBNK.index[-1], periods=future_steps+1, freq='B')[1:], conf_int_AKBNK[:, 0], conf_int_AKBNK[:, 1], color='red', alpha=0.2)
plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('Auto ARIMA Forecast for AKBNK Differenced Data')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the results for GARAN
plt.plot(differenced_data_GARAN.index, differenced_data_GARAN, label='Actual GARAN Differenced Data')
plt.plot(pd.date_range(start=differenced_data_GARAN.index[-1], periods=future_steps+1, freq='B')[1:], forecast_GARAN, label='Forecasted GARAN Differenced Data', color='red')
plt.fill_between(pd.date_range(start=differenced_data_GARAN.index[-1], periods=future_steps+1, freq='B')[1:], conf_int_GARAN[:, 0], conf_int_GARAN[:, 1], color='pink', alpha=0.2)
plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('Auto ARIMA Forecast for GARAN Differenced Data')
plt.legend()
plt.grid(True)
plt.show()






