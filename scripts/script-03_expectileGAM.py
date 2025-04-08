
# %%
#!pip install plotnine
#!pip install pytimetk

# %%
# IMPORT LIBRARIES AND DATA
#--------------------------

# system
import os

# Data Analysis
import pandas as pd
import numpy as np

# Modeling
import statsmodels.api as sm
from pygam import GAM, ExpectileGAM, l, s, f
from sklearn.preprocessing import LabelEncoder

# Visialization
import plotly.express as px
from plotnine import *
import pytimetk as tk
import textwrap

# %%
os.chdir('c:/Users/kel_m/OneDrive/Nerd_Code/PriceElasticity-2025')

os.getcwd()


# %%
# Load data
data = pd.read_csv('./data/0_raw/MyData_MnD3Items.csv')

# %%
data.info()

# %%

data.describe(include='all').T

# %%
# DATA PREPARATION
#--------------------------
selected_cols = ['MasterProductID', 'DateEnd', 'SRP']
data1 = data.groupby(selected_cols, as_index=False).agg({
    'HolidayPeriod': 'max',
    'POS_Units': 'sum',
    'POS_Amount': 'sum',
    'UnitPrice_max': 'max',
    'ARP': 'mean'
})
data1.rename(columns={'ARP': 'ARP_avg'}, inplace=True)
data1['ARP'] = np.where((~data1['POS_Units'] == 0) | (~data1['POS_Units'].isna()), data1['POS_Amount'] / data1['POS_Units'], np.nan)

# %%
data1.head()

# %%
data1.rename(columns={
    'ARP': 'price',
    'POS_Units': 'quantity_sold',
    'MasterProductID': 'product',
    'HolidayPeriod': 'is_event'	,
    'POS_Amount': 'revenue'
}, inplace=True)

data1['is_event'] = data1['is_event'].astype('bool')
data1['product'] = data1['product'].astype('str')
# %%

# EXPLORATORY DATA ANALYSIS
#--------------------------

# TRENDS: PRICE vs QUANTITY SOLD
fig = px.scatter(
    data1, 
    x='price', 
    y='quantity_sold', 
    color='product',
    opacity=0.6,
    trendline='lowess', 
    trendline_color_override='blue',
    template='none',
    title='Prodict Sales: Price vs Quantity Analysis',
    width=800,
    height=600,
).update_traces(
    marker=dict(size=7),
    hoverlabel=dict(font=dict(size=10)),
).update_layout(
    legend_title_text="Product",
    title_font=dict(size=16),
    legend_font=dict(size=10),
).update_xaxes(
    title_text='Price',
    title_font=dict(size=10),
    tickfont=dict(size=10),
).update_yaxes(
    title_text='Quantity Sold',
    title_font=dict(size=10),
    tickfont=dict(size=10),
)

fig

# %%

# EVENT ANALYSIS (OUTLIERS)
#--------------------------

fig = px.scatter(
    data1, 
    x='price', 
    y='quantity_sold', 
    color='is_event',
    facet_col='product',
    facet_col_wrap=2,
    facet_col_spacing=0.1,
    facet_row_spacing=0.1,
    opacity=0.6,
    trendline='lowess', 
    trendline_color_override='blue',
    template='none',
    title='Prodict Sales: Event Analysis',
    width=800,
    height=700,
).update_traces(
    marker=dict(size=7),
    hoverlabel=dict(font=dict(size=10)),
).update_layout(
    legend_title_text="Product",
    title_font=dict(size=16),
    legend_font=dict(size=10),
).update_xaxes(
    title_text='Price',
    title_font=dict(size=10),
    tickfont=dict(size=10),
    matches=None,
).update_yaxes(
    title_text='Quantity Sold',
    title_font=dict(size=10),
    tickfont=dict(size=10),
    matches=None,
)

for annotation in fig['layout']['annotations']:
    annotation['font']=dict(size=10)

fig.for_each_xaxis(lambda axis: axis.update(showticklabels=True))
fig.for_each_yaxis(lambda axis: axis.update(showticklabels=True))

fig

# %%

# MODELING EVENT IMPACT
#-------------------------

df_encoded = data1.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['is_event', 'product'], drop_first=False)

colnames_event = df_encoded.columns[df_encoded.columns.str.startswith('is_event')].tolist()
df_encoded[colnames_event] = df_encoded[colnames_event].astype(int)

colnames_product = df_encoded.columns[df_encoded.columns.str.startswith('product')].tolist()
df_encoded[colnames_product] = df_encoded[colnames_product].astype(int)

df_encoded

X = df_encoded[['price'] + colnames_event + colnames_product]
X = sm.add_constant(X)
y = df_encoded['quantity_sold']

model = sm.OLS(y, X).fit()

# %%
model.summary()

# %%
model.params

# %%
params_df = pd.DataFrame(model.params).T
params_df

effect_True = params_df['is_event_True'] + params_df['const'] #effect of being on a holiday period
effect_False = params_df['is_event_False'] + params_df['const']

print(effect_True)
print(effect_False)

# %%

np.log( effect_True / effect_False )

# %%
# GENERAL PRICE OPTIMIZATION: GAMs
#-------------------------

# PROBLEM: Price are non-linear
# SOLUTION: Use GAMs (Generalized Additive Models)
# NOTES:
# - GAMs are like Linear Regression, but allow for non-linear relationships
# - NOT as useful for incorporating events

# Keep the entire data as data_filtered
data_filtered = data1.query('is_event == False')
data_filtered['product'] = data_filtered['product'].astype('str')

# Create a list of unique products
unique_products = data_filtered['product'].unique()

# Create an empty dataframe to store the concatenated results
all_gam_results = pd.DataFrame()

# %%
# Loop through each product
for product in unique_products:
    # filter data for the current product
    product_data = data_filtered[data_filtered['product'] == product]

    X = product_data[['price']]
    y = product_data['quantity_sold']

    quantiles = [0.025 , 0.5 , 0.975]
    gam_results = {}

    # Fit the GAM model for the filtered data
    for q in quantiles:
        gam = ExpectileGAM(s(0), expectile=q)
        gam.fit(X, y)
        gam_results[f"pred_{q}"] = gam.predict(X)

    # Store the results in a dataframe with index that matches the original data
    predictions_gam = pd.DataFrame(gam_results).set_index(X.index)

    # Concatenate the results column-wise with the original data
    predictions_gam_df = pd.concat([product_data[['price', 'product', 'quantity_sold']], predictions_gam], axis=1)

    # Concatenate results row-wise
    all_gam_results = pd.concat([all_gam_results, predictions_gam_df], axis=0)

all_gam_results

# %%

# Visualize the GAM Price Model Results
ggplot(
    data = all_gam_results,
    mapping = aes(x='price', y='quantity_sold', color='product', group='product'),
    ) + \
    geom_ribbon(aes(ymax='pred_0.975', ymin='pred_0.025'), fill='#d3d3d3', color='#FF000000', alpha=0.75, show_legend=False) + \
    geom_point(alpha=0.5) + \
    geom_line(aes(y='pred_0.5'), color='blue') + \
    facet_wrap('product', scales='free') + \
    labs(title='GAM Price vs Quantity Model') + \
    scale_color_manual(values=list(tk.palette_timetk().values())) + \
    tk.theme_timetk(width=800, height=600)

# %%

# Optimize Price for Predicted Revenue
for col in all_gam_results.columns:
    if col.startswith('pred'):
        all_gam_results['revenue_' + col] = all_gam_results['price'] * all_gam_results[col]

all_gam_results['revenue_actual'] = all_gam_results['price'] * all_gam_results['quantity_sold']

all_gam_results

# %%
best_50 = all_gam_results \
    .groupby('product') \
    .apply(lambda x: x[x['revenue_pred_0.5'] == x['revenue_pred_0.5'].max()].head(1)) \
    .reset_index(level=0, drop=True)

best_975 = all_gam_results \
    .groupby('product') \
    .apply(lambda x: x[x['revenue_pred_0.975'] == x['revenue_pred_0.975'].max()].head(1)) \
    .reset_index(level=0, drop=True)

best_025 = all_gam_results \
    .groupby('product') \
    .apply(lambda x: x[x['revenue_pred_0.025'] == x['revenue_pred_0.025'].max()].head(1)) \
    .reset_index(level=0, drop=True)

# %%
# Visualize thr GAM Revenue Optimization Results
(
   ggplot(
        data = all_gam_results,
        mapping = aes(x='price', y='revenue_pred_0.5', color='product', group='product'),
        ) + \
        geom_ribbon(aes(ymax='revenue_pred_0.975', ymin='revenue_pred_0.025'), fill='#d3d3d3', color='#FF000000', alpha=0.75, show_legend=False) + \
        # Uncomment to add actual revenue points
        geom_point(aes(y='revenue_actual'), alpha=0.15, color='#2C3D50') + \
        geom_line(aes(y='revenue_pred_0.5'), alpha=0.5) + \
        geom_point(data=best_50, color='red') + \
        geom_point(data=best_975, mapping=aes(y='revenue_pred_0.975'), color='blue') + \
        geom_point(data=best_025, mapping=aes(y='revenue_pred_0.025'), color='blue') + \
        facet_wrap('product', scales='free') + \
        labs(
            title='Price Optimization',
            subtitle='Maximum Mediam Revenue (red) vs 95% Maximum Confidence Interval (blue)',
            x='Price',
            y='Predicted Revenue'
            ) + \
        scale_color_manual(values=list(tk.palette_timetk().values())) + \
        tk.theme_timetk(width=800, height=600) 
)

# %% 
best_50[['product', 'price', 'revenue_pred_0.5', 'revenue_pred_0.025', 'revenue_pred_0.975']]


# %%

# MODELING GAMS WITH EVENTS:
#-------------------------
# Essentially the same process as above, but we need to filter out the "No Promo" events
# Gets a little tricky because of limited data points for each event

# Keep the entire data as data_filtered
data_filtered = data1.query('is_event >= 0 ') # != "No Promo"

# Create a list od unique products and events
unique_products = data_filtered['product'].unique()

events_only_gam_results = pd.DataFrame()

# Loop through each product
for product in unique_products:

    # Filter data for current product and event
    product_event_data = data_filtered[(data_filtered['product'] == product)]

    if len(product_event_data) == 0:
        continue # skip to the next iteration if no data for current product-event combination

    X = product_event_data[['price', 'is_event']]
    y = product_event_data['quantity_sold']

    # NEW: Encode the event column
    le = LabelEncoder()
    X['is_event'] = le.fit_transform(X['is_event'])

    # NEW: use f(1) to indicate that the event column is categorical
    gam = GAM(l(0) + f(1))

    gam.fit(X, y)
    gam_results["pred_0.025"] = gam.predict(X)
    gam_results["pred_0.5"] = gam.predict(X)
    gam_results["pred_0.975"] = gam.predict(X)

    # Store the results in a dataframe with index that matches the original data
    predictions_gam = pd.DataFrame(gam_results).set_index(X.index)

    # Concatenate the results column-wise with original data
    predictions_gam_df = pd.concat([product_event_data[['price', 'product', 'is_event', 'quantity_sold']], predictions_gam], axis=1)

    # Concatenate results row-wise
    events_only_gam_results = pd.concat([events_only_gam_results , predictions_gam_df], axis=0)

events_only_gam_results

# %%
# Visualize the GAM Price Model Results

# Ensure 'product' and 'is_event' are categorical
events_only_gam_results['product'] = events_only_gam_results['product'].astype('category')
events_only_gam_results['is_event'] = events_only_gam_results['is_event'].astype('category')

ggplot(
    data = events_only_gam_results,
    mapping = aes(x='price', y='quantity_sold', color='product')
) + \
    geom_point(alpha=0.5) + \
    geom_line(aes(y = 'pred_0.5'), color='blue') + \
    facet_grid('product ~ is_event', scales='free') + \
    labs(title='[Special Events] GAM Price vs Quantity Model') + \
    scale_color_manual(values = list(tk.palette_timetk().values())) + \
    tk.theme_timetk(width=800, height=600)

# %%
# THEN OPTIMIZE PRICE FOR PREDICTED DAILY REVENUE

# Optimixation Price for Predicted Daily Revenue

for col in events_only_gam_results.columns:
    if col.startswith('pred'):
        events_only_gam_results['revenue_' + col] = events_only_gam_results['price'] + events_only_gam_results[col]

events_only_gam_results

# %%
best_50 = events_only_gam_results \
    .groupby(['product', 'is_event']) \
    .apply(lambda x: x[x['revenue_pred_0.5'] == x['revenue_pred_0.5'].max()].head(1)) \
    .reset_index(level=0, drop=True)

# %%
# Visualize the GAM Revenue Optimization Results    

# Define the wrap function
def wrap_label(label, width=10):
    return '\n'.join(textwrap.wrap(label, width=width))

# %%

ggplot(
    data = events_only_gam_results,
    mapping = aes(x='price', y='revenue_pred_0.5', color='product'), 
) + \
    geom_line(alpha=0.5) + \
    geom_point(aes(x='price', y='revenue_pred_0.5', color='product'), 
               data=events_only_gam_results, size=0.2) + \
    geom_point(data = best_50 , color='red') + \
    facet_grid('product ~ is_event', scales='free', labeller=labeller(product=wrap_label)) + \
    labs(
        title = '[Special Events] Price Optimization',
        subtitle = 'Maximum Median Revenue (red)',
        x = 'Price',
        y = 'Predicted Revenue'
    ) + \
    scale_color_manual(values = list(tk.palette_timetk().values())) + \
    tk.theme_timetk(width=800, height=600) + \
    theme(strip_text_y=element_text(size=6))

# %%
best_50

# BUSINESS INSIGHTS:
#------------------
# Events have a significant inpact on price optimization
# Prices should be optimize for each special event (e.g. Black Friday, Christmas)
# For demand decreasing events (e.g. New Iphone Model comes out), prices hould be lowered to maximize revenue based on historical data