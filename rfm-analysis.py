import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import missingno as msno
import warnings
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

data = pd.read_csv('/content/drive/MyDrive/Магистратура/2 курс/СиТАМ/Датасеты/Scanner data.csv')

def indexes(data):
  str = ''
  ind = 1
  for el in data:
    str += '{}{}'.format(el, ', ')
    if ind % 5 == 0:
      str += '\n\t\t  '
    ind += 1
  return str[:-2]

data

data.info()

data = data.drop('Unnamed: 0', axis=1)

data.insert(6, 'Price', (data['Sales_Amount'] / data['Quantity']).round(2))

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Date'] = data['Date'].dt.strftime('%d.%m.%Y')
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')

row, col = data.shape
print('Количество записей в наборе =', row)
print('Количество признаков в наборе =', col)

print('Список признаков:', indexes(data.columns))

duplicates = data[data.duplicated()]

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Price'], label='Original Data', color='mediumturquoise')
plt.scatter(duplicates.index, duplicates['Price'], color='coral', label='Duplicates')
plt.title('Visualization of duplicate rows')
plt.xlabel('Number of observation')
plt.ylabel('Price')
plt.legend()
plt.show()

print('Количество покупателей = %d' % data['Customer_ID'].nunique())
print('Количество чеков = %d' % data['Transaction_ID'].nunique())
print('Количество категорий товаров = %d' % data['SKU_Category'].nunique())
print('Количество товаров = %d' % data['SKU'].nunique())

categories = data.groupby('SKU_Category').agg({
    'SKU': 'count',
    'Quantity': 'sum'
}).reset_index()
categories.sort_values(by='SKU')

other_sku = categories[categories['SKU'] <= 2000]['SKU'].sum()
other_q = categories[categories['SKU'] <= 2000]['Quantity'].sum()
other_row = ['Other categories', other_sku, other_q]

categories = categories.drop(categories[categories['SKU'] <= 2000].index)
categories.loc[len(categories)] = other_row
categories

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
colors = sns.color_palette('GnBu_r', len(categories['SKU_Category']))

axes.pie(categories['SKU'], labels=categories['SKU_Category'], colors=colors, startangle=30, autopct='%1.1f%%')
axes.set_title('By categories')

plt.tight_layout()
plt.show()

transactions = data.groupby('Customer_ID')['Transaction_ID'].nunique().reset_index().sort_values(by='Transaction_ID')
transactions.tail(10)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

dt = transactions.tail(10)
dt['Customer_ID'] = dt['Customer_ID'].apply(lambda x : str(x))
axes.barh(dt['Customer_ID'], dt['Transaction_ID'], color='mediumturquoise')
for index, value in enumerate(dt['Transaction_ID']):
    axes.annotate(value, xy=(value+1, index-0.1), color='slategray')

axes.set_title('Buyers with the most transactions')
axes.set_ylabel('Customer')
axes.set_xlabel('Quantity')

plt.tight_layout()
plt.show()

revenue_daily = data.groupby('Date')['Sales_Amount'].sum().reset_index(name='Revenue')
temp = revenue_daily.copy()
temp['Month'] = temp['Date'].dt.to_period('M')
temp
revenue_monthly = temp.groupby('Month')['Revenue'].sum().reset_index(name='Revenue')

display(revenue_daily)
display(revenue_monthly)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(13, 7))
colors = sns.color_palette('GnBu', 4)

axes[0].plot(revenue_daily['Date'], revenue_daily['Revenue'], color='mediumturquoise')
axes[1].plot(revenue_monthly['Month'].apply(lambda x : str(x)), revenue_monthly['Revenue'], color='mediumturquoise')

axes[0].set_title('Daily revenue')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Revenue')

axes[1].set_title('Monthly revenue')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Revenue')

plt.tight_layout()
plt.show()

data.groupby(['SKU_Category', 'SKU'])['Price'].nunique().sort_values()

data[data['SKU'] == 'FZM7U']

label_encoder = LabelEncoder()

dcorr = data[['Date', 'Quantity', 'Price', 'Sales_Amount']]
dcorr['Category'] = label_encoder.fit_transform(data['SKU_Category'])
dcorr.corr().style.background_gradient(cmap = 'GnBu')

today = data['Date'].max() + timedelta(days=1)
print('Last date - %s' % data['Date'].max())
print('Next date - %s' % today)

rfm = data.groupby(['Customer_ID']).agg({
    'Date' : lambda x : (today - x.max()).days,
    'Transaction_ID' : 'count',
    'Sales_Amount' : 'sum'
})

rfm.rename(columns=({
    'Date' : 'Recency',
    'Transaction_ID' : 'Frequency',
    'Sales_Amount' : 'Monetary'}), inplace=True)

rfm

rfm['Monetary'].max()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
colors = sns.color_palette('GnBu', 4)

sns.distplot(rfm['Recency'], ax=axes[0], color='mediumturquoise')
sns.distplot(rfm['Frequency'], ax=axes[1], color='mediumturquoise')
sns.distplot(rfm['Monetary'], ax=axes[2], color='mediumturquoise')

plt.tight_layout()
plt.show()

r_label = range(3, 0, -1)
f_label = range(1, 4)
m_label = range(1, 4)

rfm['R'] = pd.qcut(rfm['Recency'], q=3, labels=r_label)
rfm['F'] = pd.qcut(rfm['Frequency'], q=3, labels=f_label)
rfm['M'] = pd.qcut(rfm['Monetary'], q=3, labels=m_label)

rfm

rfm['RFM'] = rfm.apply(lambda x : str(int(x['R'])) + str(int(x['F'])) + str(int(x['M'])), axis=1)
rfm

rfm['Score'] = rfm[['R', 'F', 'M']].sum(axis=1)
rfm

rfm.loc[(rfm['RFM'] == '111'), 'Status'] = '9 Lost'
rfm.loc[(rfm['RFM'].str.contains(r'11[23]')), 'Status'] = '8 Big Spenders'
rfm.loc[(rfm['RFM'].str.contains(r'1[23][23]')), 'Status'] = '7 Needs attention'
rfm.loc[(rfm['RFM'].str.contains(r'1[23]1')), 'Status'] = '6 Frequent Customers'
rfm.loc[(rfm['RFM'].str.contains(r'[23]1[23]')), 'Status'] = '5 Infrequent High Spenders'
rfm.loc[(rfm['RFM'].str.contains(r'[23]11')), 'Status'] = '4 Recent Customers'
rfm.loc[(rfm['RFM'].str.contains(r'[23][23]1')), 'Status'] = '3 Frequent Low Spenders'
rfm.loc[(rfm['RFM'].str.contains(r'[23][23][23]')), 'Status'] = '2 Loyal'
rfm.loc[(rfm['RFM'] == '333'), 'Status'] = '1 VIP'

rfm

result = rfm.groupby(['Status']).agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'Monetary' : ['mean', 'sum', 'count']
})

result

groups = rfm.groupby('Status').size().reset_index(name='Count').sort_values(by='Count')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

axes.barh(groups['Status'], groups['Count'], color='mediumturquoise')
for index, value in enumerate(groups['Count']):
    axes.annotate(value, xy=(value+30, index-0.1), color='slategray')

plt.tight_layout()
plt.show()
