import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

# Using uploaded files
file_name = 'TCS_scaled.csv'

# the filename of the uploaded file
file_name = 'dataset_inflation.csv'

# Read the CSV file
df = pd.read_csv(file_name, index_col='year', parse_dates=True)
print(df.head())

df.plot(figsize=(12,6))

#statistical analysis
# mean median std min max 2,3,4 quartile
df.describe()

#skewness & kurtosis
import pandas as pd
import numpy as np

# Extract columns as numpy arrays
x = df["inflation rate (USA)"].to_numpy()
freq = 1

# Total frequency
size = 11

# Mean
xbar = np.sum(x * freq) / size
print("Mean (x̄):", xbar)

# Variance
vr = np.sum(((x - xbar) ** 2) * freq) / size
print("Variance:", vr)

# Standard deviation
sdev = np.sqrt(vr)
print("Standard Deviation:", sdev)

# Third moment (m3)
m3 = np.sum(((x - xbar) ** 3) * freq) / size
print("Third moment (m3):", m3)

# Fourth moment (m4)
m4 = np.sum(((x - xbar) ** 4) * freq) / size
print("Fourth moment (m4):", m4)

# Skewness (a3)
a3 = m3 / (sdev ** 3)
print("Skewness (a3):", a3)

# Kurtosis (a4)
a4 = m4 / (sdev ** 4)
print("Kurtosis (a4):", a4)

# Optional: interpret results
if a3 > 0:
    print("→ Distribution is positively skewed (right-skewed).")
elif a3 < 0:
    print("→ Distribution is negatively skewed (left-skewed).")
else:
    print("→ Distribution is symmetric.")

if a4 > 3:
    print("→ Leptokurtic (peaked distribution).")
elif a4 < 3:
    print("→ Platykurtic (flatter distribution).")
else:
    print("→ Mesokurtic (normal-like distribution).")
