import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


import pandas as pd
import numpy as np
import glob

# Path to the directory containing your CSV files
path = 'C:\\Users\EylülRanaSaraç\\OneDrive - boun.edu.tr\\Masaüstü\\IE 423\\Project Part 2\\golden-girlz\\ProjectPart2\\20180101_20231121_bist30'
all_files = glob.glob(path + "/*_bist30.csv")

# Create a list to hold dataframes
dfs = []

# Loop through the files and read them into individual dataframes
for filename in all_files:
    df = pd.read_csv(filename, parse_dates=['timestamp'])
    dfs.append(df)

# Concatenate all dataframes into one
full_df = pd.concat(dfs, ignore_index=True)

# Pivot the dataframe to have timestamps as rows and stocks as columns
pivot_df = full_df.pivot(index='timestamp', columns='short_name', values='price')

# Calculate the correlation matrix
correlation_matrix = pivot_df.corr()

# Filter out pairs with correlation below a certain threshold
threshold = 0.9  # Example threshold, adjust as needed
high_correlation_pairs = correlation_matrix[(correlation_matrix >= threshold) & (correlation_matrix < 1.0)]


pd.set_option('display.max_rows', 1000)  # or set a specific large number


# Display the highly correlated pairs
#print(high_correlation_pairs.stack())
#high_correlation_pairs.to_csv('high_correlation_pairs.csv')

sorted_pairs = correlation_matrix.unstack().sort_values(ascending=False)

# Kendi kendine korelasyonları (1.0) ve tekrar eden çiftleri çıkarın
unique_sorted_pairs = sorted_pairs.drop_duplicates()
non_self_pairs = unique_sorted_pairs[unique_sorted_pairs < 1]

# En yüksek 20 çifti alın
top_20_pairs = non_self_pairs.head(20)
print(top_20_pairs)