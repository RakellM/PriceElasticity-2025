#!/usr/bin/env python3
"""Price Elasticity Analysis
Using Kaggle's Data

Install:
- polars
- pyarrow
- pandas
"""
__version__ = "0.1.0"
__author__ = "Raquel Marques"
__license__ = "Unlicense"

# %%
# Libraries
#import polars as pl
#import pyarrow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Import Data - raw
#dsStoreSales = pl.read_parquet('./data/0_raw/Store_Sales_Price_Elasticity_Promotions_Data.parquet')
dsStoreSales = pd.read_parquet('./data/0_raw/Store_Sales_Price_Elasticity_Promotions_Data.parquet', engine='pyarrow')

print(dsStoreSales.head()) # print first 5 rows

# Understanding data
## Data Info
print("\nDataset information:")
print(dsStoreSales.info())

#-- Convert Sold_Date to datetime
dsStoreSales['Sold_Date'] = pd.to_datetime(dsStoreSales['Sold_Date']).dt.date

#-- Check for missing values
print("\nCheck for missing values:")
print(dsStoreSales.isnull().sum())

#-- Descriptive statistics
print("\nDescriptive Statistics:")
print(dsStoreSales.describe())

# Histograms
dsStoreSales[['Qty_Sold', 'Total_Sale_Value']].hist(bins=20, figsize=(10, 5))
plt.suptitle("Distribution of Quantity Sold and Sales Value")
plt.show()

# Boxplots to check for outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=dsStoreSales[['Qty_Sold', 'Total_Sale_Value']])
plt.title("Boxplot of Quantity Sold and Sales Value")
plt.show()

# Sales trends over time
plt.figure(figsize=(12, 6))
dsStoreSales.groupby('Sold_Date')['Total_Sale_Value'].sum().plot()
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Sales Trend Over Time")
plt.xticks(rotation=45)
plt.show()


# Price Elasticity of Demand (PED) Analysis


# Calculate unit price
dsStoreSales['Unit_Price'] = dsStoreSales['Total_Sale_Value'] / dsStoreSales['Qty_Sold']

# Aggregate data: total quantity & avg price per SKU per date
sku_daily_sales = dsStoreSales.groupby(['SKU_Coded', 'Sold_Date']).agg(
    {'Qty_Sold': 'sum', 'Unit_Price': 'mean'}
).reset_index()

# Compute percentage change in quantity & price
sku_daily_sales['Pct_Change_Qty'] = sku_daily_sales.groupby('SKU_Coded')['Qty_Sold'].pct_change()
sku_daily_sales['Pct_Change_Price'] = sku_daily_sales.groupby('SKU_Coded')['Unit_Price'].pct_change()

# Compute price elasticity
sku_daily_sales['Price_Elasticity'] = sku_daily_sales['Pct_Change_Qty'] / sku_daily_sales['Pct_Change_Price']

# Drop NaN values (first row for each SKU will have NaN due to pct_change())
sku_daily_sales = sku_daily_sales.dropna()

# Display results
print(sku_daily_sales[['SKU_Coded', 'Sold_Date', 'Qty_Sold', 'Unit_Price', 'Price_Elasticity']].head())

# Summary statistics of price elasticity
print(sku_daily_sales['Price_Elasticity'].describe())


## Scatter Plot: Price vs. Quantity Sold

plt.figure(figsize=(10, 6))
sns.scatterplot(data=sku_daily_sales, x='Unit_Price', y='Qty_Sold', alpha=0.5)
plt.xlabel("Unit Price")
plt.ylabel("Quantity Sold")
plt.title("Price vs. Quantity Sold")
plt.show()


## Elasticity by Promo Status

# Merge promo info back into the dataset
dsStoreSales['Unit_Price'] = dsStoreSales['Total_Sale_Value'] / dsStoreSales['Qty_Sold']
merged_df = sku_daily_sales.merge(
    dsStoreSales[['SKU_Coded', 'Sold_Date', 'On_Promo']].drop_duplicates(), 
    on=['SKU_Coded', 'Sold_Date'], 
    how='left'
)

# Boxplot of price elasticity by promo status
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_df, x='On_Promo', y='Price_Elasticity')
plt.xlabel("On Promotion (0 = No, 1 = Yes)")
plt.ylabel("Price Elasticity")
plt.title("Price Elasticity by Promo Status")
plt.show()

## Categorizing Demand Elasticity

# Classify elasticity
def categorize_elasticity(value):
    if value < -1:
        return "Elastic"
    elif -1 <= value <= 0:
        return "Inelastic"
    elif value > 0:  # Some edge cases might have positive elasticity
        return "Unusual (+ve)"
    return "Unitary"

sku_daily_sales['Elasticity_Category'] = sku_daily_sales['Price_Elasticity'].apply(categorize_elasticity)

# Count occurrences of each category
print(sku_daily_sales['Elasticity_Category'].value_counts())

# Plot distribution of elasticity categories
plt.figure(figsize=(8, 5))
sns.countplot(data=sku_daily_sales, x='Elasticity_Category', order=['Elastic', 'Inelastic', 'Unusual (+ve)'])
plt.xlabel("Elasticity Category")
plt.ylabel("Count of SKU-Days")
plt.title("Distribution of Price Elasticity Categories")
plt.show()

## Elasticity by Product Class

### Aggregate Elasticity by Product Class

# Merge product class info
sku_daily_sales = sku_daily_sales.merge(
    dsStoreSales[['SKU_Coded', 'Product_Class_Code']].drop_duplicates(), 
    on='SKU_Coded', 
    how='left'
)

# Compute average elasticity per product class
product_class_elasticity = sku_daily_sales.groupby('Product_Class_Code')['Price_Elasticity'].mean().reset_index()

# Display results
print(product_class_elasticity.sort_values(by='Price_Elasticity', ascending=True))

### Boxplot: Elasticity by Product Class

plt.figure(figsize=(12, 6))
sns.boxplot(data=sku_daily_sales, x='Product_Class_Code', y='Price_Elasticity')
plt.xlabel("Product Class Code")
plt.ylabel("Price Elasticity")
plt.title("Price Elasticity by Product Class")
plt.xticks(rotation=45)
plt.show()

### Bar Chart: Elasticity Categories by Product Class

# Count elasticity categories by product class
elasticity_by_class = sku_daily_sales.groupby(['Product_Class_Code', 'Elasticity_Category']).size().unstack(fill_value=0)

# Normalize counts to show proportions
elasticity_by_class = elasticity_by_class.div(elasticity_by_class.sum(axis=1), axis=0)

# Plot stacked bar chart
elasticity_by_class.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='coolwarm')
plt.xlabel("Product Class Code")
plt.ylabel("Proportion of Elasticity Categories")
plt.title("Distribution of Price Elasticity Categories by Product Class")
plt.legend(title="Elasticity Category")
plt.xticks(rotation=45)
plt.show()

## Seasonality Analysis

### Add Time Features (Month, Weekday, Season)

# Extract month and weekday
sku_daily_sales['Month'] = pd.to_datetime(sku_daily_sales['Sold_Date']).dt.month
sku_daily_sales['Weekday'] = pd.to_datetime(sku_daily_sales['Sold_Date']).dt.day_name()

# Define seasons (Northern Hemisphere)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

sku_daily_sales['Season'] = sku_daily_sales['Month'].apply(get_season)

### Monthly Trends: Price Elasticity Over Time

plt.figure(figsize=(12, 6))
sns.boxplot(data=sku_daily_sales, x='Month', y='Price_Elasticity')
plt.xlabel("Month")
plt.ylabel("Price Elasticity")
plt.title("Monthly Price Elasticity Trends")
plt.show()

### Weekly Trends: Elasticity by Weekday

plt.figure(figsize=(10, 5))
sns.boxplot(data=sku_daily_sales, x='Weekday', y='Price_Elasticity', order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
plt.xlabel("Weekday")
plt.ylabel("Price Elasticity")
plt.title("Price Elasticity by Day of the Week")
plt.xticks(rotation=45)
plt.show()

### Seasonal Trends: Comparing Elasticity Across Seasons

plt.figure(figsize=(10, 5))
sns.boxplot(data=sku_daily_sales, x='Season', y='Price_Elasticity', order=['Winter', 'Spring', 'Summer', 'Fall'])
plt.xlabel("Season")
plt.ylabel("Price Elasticity")
plt.title("Price Elasticity by Season")
plt.show()


## Geographic Differences

### Store-Level Elasticity Analysis

# Merge store info
sku_daily_sales = sku_daily_sales.merge(dsStoreSales[['SKU_Coded', 'Store_Number']].drop_duplicates(), 
                                        on='SKU_Coded', how='left')

# Compute avg elasticity per store
store_elasticity = sku_daily_sales.groupby('Store_Number')['Price_Elasticity'].mean().reset_index()

# Display results
print(store_elasticity.sort_values(by='Price_Elasticity', ascending=True))

### Elasticity Distribution by Store

plt.figure(figsize=(12, 6))
sns.histplot(store_elasticity['Price_Elasticity'], bins=30, kde=True)
plt.xlabel("Price Elasticity")
plt.ylabel("Count of Stores")
plt.title("Distribution of Price Elasticity Across Stores")
plt.show()

### Store-Level Boxplot: Elasticity by Region

plt.figure(figsize=(12, 6))
sns.boxplot(data=sku_daily_sales, x='Store_Number', y='Price_Elasticity')
plt.xlabel("Store Number")
plt.ylabel("Price Elasticity")
plt.title("Price Elasticity by Store")
plt.xticks(rotation=90)
plt.show()




######

## Uniques
### Stores
dsStoreNumber = dsStoreSales['Store_Number'].unique()

#print("\nStore Number list:")
#print(dsStoreNumber)
print(f"\nStore Number unique count: {len(dsStoreNumber)}")

### SKU
dsSKU = dsStoreSales['SKU_Coded'].unique()

#print("\nSKU list:")
#print(dsSKU)
print(f"\nSKU unique count: {len(dsSKU)}")

### Product Class Code
dsClass = dsStoreSales['Product_Class_Code'].unique()

#print("\nClass list:")
#print(dsClass)
print(f"\nClass unique count: {len(dsClass)}")





#-- What do we need for a price elasticity analysis?
#-- 
#summary1 = dsStoreSales.group_by("Store_Number").agg(pl.col("Total_Sale_Value").sum()).sort("Total_Sale_Value")

#print(summary1)




