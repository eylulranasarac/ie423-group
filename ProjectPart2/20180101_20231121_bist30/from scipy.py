from scipy.stats import norm

z_score = [0.70,
0.68,
-0.17,
-0.42,
0.20,
-0.10,
0.75,
0.24,
0.25,
-1.41,
-0.61,
-1.07,
-1.04]
percentile = norm.cdf(z_score) * 100
print(f"Approximately {percentile}%")
