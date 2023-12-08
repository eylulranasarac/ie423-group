import glob
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import normaltest
import pmdarima as pm

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
plt.figure(figsize=(8, 4))
plt.plot(differenced_data_AKBNK.index, differenced_data_AKBNK)
plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('Differenced AKBNK Price Over Time')
#plt.show()

# Plotting the differenced data for GARAN
plt.figure(figsize=(8, 4))
plt.plot(differenced_data_GARAN.index, differenced_data_GARAN)
plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('Differenced GARAN Price Over Time')
#plt.show()

order_AKBNK = (1, 1, 1)
model_AKBNK = SARIMAX(differenced_data_AKBNK, order=order_AKBNK)
model_AKBNK_fit = model_AKBNK.fit()

# Print the fitted model summary for AKBNK
print(model_AKBNK_fit.summary())

# Make predictions for AKBNK
steps_AKBNK = 10
forecast_AKBNK = model_AKBNK_fit.get_forecast(steps=steps_AKBNK)
forecast_index_AKBNK = pd.date_range(start=combined_data_AKBNK.index[-1], periods=steps_AKBNK + 1, freq='B')[1:]

# Plotting the differenced AKBNK data
plt.plot(differenced_data_AKBNK.index, differenced_data_AKBNK, label='Differenced AKBNK Data')

# Plotting the AKBNK forecast
plt.plot(forecast_index_AKBNK, forecast_AKBNK.predicted_mean.values, label='AKBNK Forecast', linestyle='dashed')

# Plotting AKBNK confidence intervals
conf_int_AKBNK = forecast_AKBNK.conf_int()
plt.fill_between(forecast_index_AKBNK, conf_int_AKBNK['lower price'], conf_int_AKBNK['upper price'], alpha=0.2)

plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('AKBNK Differenced Price Forecast (Manual ARIMA)')
plt.legend()
#plt.show()

order_GARAN = (1, 1, 1)
model_GARAN = SARIMAX(differenced_data_GARAN, order=order_GARAN)
model_GARAN_fit = model_GARAN.fit()

# Print the fitted model summary for GARAN
print(model_GARAN_fit.summary())

# Make predictions for GARAN
steps_GARAN = 10
forecast_GARAN = model_GARAN_fit.get_forecast(steps=steps_GARAN)
forecast_index_GARAN = pd.date_range(start=combined_data_GARAN.index[-1], periods=steps_GARAN + 1, freq='B')[1:]

# Plotting the differenced GARAN data
plt.plot(differenced_data_GARAN.index, differenced_data_GARAN, label='Differenced GARAN Data')

# Plotting the GARAN forecast
plt.plot(forecast_index_GARAN, forecast_GARAN.predicted_mean.values, label='GARAN Forecast', linestyle='dashed')

# Plotting GARAN confidence intervals
conf_int_GARAN = forecast_GARAN.conf_int()
plt.fill_between(forecast_index_GARAN, conf_int_GARAN['lower price'], conf_int_GARAN['upper price'], alpha=0.2)

plt.xlabel('Timestamp')
plt.ylabel('Differenced Price')
plt.title('GARAN Differenced Price Forecast (Manual ARIMA)')
plt.legend()
#plt.show()

# Residuals for AKBNK
residuals_AKBNK = differenced_data_AKBNK - model_AKBNK_fit.fittedvalues

# Residuals for GARAN
residuals_GARAN = differenced_data_GARAN - model_GARAN_fit.fittedvalues


# ADF test for residuals of AKBNK
result_residuals_AKBNK = adfuller(residuals_AKBNK)
p_value_residuals_AKBNK = result_residuals_AKBNK[1]
if p_value_residuals_AKBNK <= 0.05:
    print('The residuals of AKBNK are likely stationary.')
else:
    print('The residuals of AKBNK are likely non-stationary.')

# ADF test for residuals of GARAN
result_residuals_GARAN = adfuller(residuals_GARAN)
p_value_residuals_GARAN = result_residuals_GARAN[1]
if p_value_residuals_GARAN <= 0.05:
    print('The residuals of GARAN are likely stationary.')
else:
    print('The residuals of GARAN are likely non-stationary.')


# Normality test for AKBNK residuals
_, p_value_normal_AKBNK = normaltest(residuals_AKBNK)
if p_value_normal_AKBNK <= 0.05:
    print('The residuals of AKBNK are not normally distributed.')
else:
    print('The residuals of AKBNK are normally distributed.')

# Normality test for GARAN residuals
_, p_value_normal_GARAN = normaltest(residuals_GARAN)
if p_value_normal_GARAN <= 0.05:
    print('The residuals of GARAN are not normally distributed.')
else:
    print('The residuals of GARAN are normally distributed.')

