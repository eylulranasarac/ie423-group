from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt

# pytrends oturumunu başlat
pytrend = TrendReq()

# Google Trends için sorgu ayarları
kw_list = ["PETKM"]  # Örnek anahtar kelime listesi
timeframe = '2018-09-01 2018-10-01'  # Belirli bir tarih aralığı
pytrend.build_payload(kw_list, timeframe=timeframe, geo='TR')

# Verileri al
trends_data = pytrend.interest_over_time()

# # Gereksiz sütunları sil (örneğin, 'isPartial')
# trends_data = trends_data.drop(labels=['isPartial'], axis='columns')

# Veriyi kontrol et
print(trends_data.head())
company_name = kw_list[0]
# Grafik çiz
plt.figure(figsize=(10, 5))
plt.plot(trends_data.index, trends_data[kw_list[0]])
plt.title('Monthly Search Amount of ' + str(company_name) +' in Turkey')
plt.xlabel('Date')
plt.ylabel(str(company_name) + ' Search')
plt.tight_layout()  # Grafiğin düzgün görünmesi için
plt.show()
save_path = "PETKM SP" # Hisseye göre farklı klasör seç
year_month = timeframe[:7]
save_name = f"{kw_list[0]} {year_month}"
plt.savefig(f"{save_path}/{save_name}.png", format='png')


