# %%
#!pip install pygam

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import s, ExpectileGAM

# %%

# DATA GENERATION
#----------------

# Data Generation
np.random.seed(0)
n = 100
price = np.sort(np.random.exponential(scale=100, size=n))
quantity = 1000 - 5 * price + np.random.normal(loc=0, size=n, scale=50)
quantity = quantity.clip(min=0)

# Add outliers
n_outliers = 10
outlier_prices = np.random.uniform(5, 50, n_outliers)
outlier_quantity = 1100 + np.random.normal(loc=0, scale=50, size=n_outliers)
price = np.concatenate([price, outlier_prices])
quantity = np.concatenate([quantity, outlier_quantity])

# Add outliers
n_outliers = 10
outlier_prices = np.random.uniform(51, 100, n_outliers)
outlier_quantity = 900 + np.random.normal(loc=0, scale=50, size=n_outliers)
price = np.concatenate([price, outlier_prices])
quantity = np.concatenate([quantity, outlier_quantity])

df = pd.DataFrame({
    'Price': price, 
    'Quantity': quantity
})

# Filter out prices less than 5
df = df[df['Price'] >= 5]

# %% 

# MODELING
#----------------

# Reshape data
X = df[['Price']]
y = df['Quantity']

# Quantile GAMs
quantiles = [0.025, 0.5, 0.975]
gam_results = {}

for q in quantiles:
    gam = ExpectileGAM(s(0), expectile=q)
    gam.fit(X, y)
    gam_results[q] = gam

gam_results

# %%

# VISUALIZATION
#----------------

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Price'], df['Quantity'], alpha=0.5, label='Data Points')

# Plot Quantile GAMs
XX = np.linspace(df['Price'].min(), df['Price'].max(), 1000).reshape(-1, 1)
for q, gam in gam_results.items():
    plt.plot(XX, gam.predict(XX), label=f"{int(q * 100)}th Quantile GAM") 

# Add title and labels
plt.xlabel("Price")
plt.ylabel("Quantity Demanded")
plt.title("Quantile GAMs on Price Elasticity of Demand (Outliers Removed)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# EXPECTILE GAM BENEFITS:
#----------------
# 1. Quantile GAMS take into account changing variance in the data.
# 2. Allow us to take into account the best and worst case scenarios in the Price Optimization.
