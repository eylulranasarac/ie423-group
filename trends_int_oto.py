from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# pytrends oturumunu başlat
pytrend = TrendReq(hl='en-US', tz=360)

# Başlangıç ve bitiş tarihlerini ayarla
start_date = datetime(2017, 6, 1)
end_date = datetime(2019, 7, 1)

# Tüm verileri saklamak için boş bir DataFrame oluştur
all_trends = pd.DataFrame()

while start_date < end_date:
    # Sonraki ayın ilk gününü al
    next_month = start_date + timedelta(days=31)
    next_month = next_month.replace(day=1)

    # Tarih aralığını formatla
    timeframe = start_date.strftime('%Y-%m-%d') + ' ' + next_month.strftime('%Y-%m-%d')

    try:
        # Google Trends için sorgu ayarları
        pytrend.build_payload(kw_list=['PETKM'], timeframe=timeframe, geo='TR')

        # Verileri al
        trends_data = pytrend.interest_over_time()

        # Gereksiz sütunları sil (örneğin, 'isPartial')
        trends_data = trends_data.drop(labels=['isPartial'], axis='columns', errors='ignore')

        # Verileri birleştir
        if not trends_data.empty:
            all_trends = pd.concat([all_trends, trends_data])

    except pytrend.exceptions.TooManyRequestsError:
        # Eğer çok fazla istek hatası alınırsa, biraz bekleyin
        print("Too many requests, sleeping for 60 seconds.")
        time.sleep(60)  # 60 saniye bekle
        continue  # Döngünün başına dön

    # Google Trends için sorgu ayarları
    pytrend.build_payload(kw_list=['PETKM'], timeframe=timeframe, geo='TR')

    # Verileri al
    trends_data = pytrend.interest_over_time()

    # Gereksiz sütunları sil (örneğin, 'isPartial')
    trends_data = trends_data.drop(labels=['isPartial'], axis='columns', errors='ignore')

    # Verileri birleştir
    if not trends_data.empty:
        all_trends = pd.concat([all_trends, trends_data])

    # Başlangıç tarihini sonraki aya taşı
    start_date = next_month

# Veriyi kontrol et
print(all_trends.head())

# Grafik çiz
plt.figure(figsize=(15, 5))
plt.plot(all_trends.index, all_trends['PETKM'])
plt.title('Monthly Search Amount of PETKM in Turkey')
plt.xlabel('Days')
plt.ylabel('PETKM Search')

# Grafiği kaydet
plt.savefig('/mnt/data/google_trends_petkm.png')

# Grafiği göster
plt.show()
