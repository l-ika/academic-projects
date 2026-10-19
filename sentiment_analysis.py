import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
import missingno as msno
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Rotten tomatoes movie reviews.csv')

def indexes(data):
  str = ''
  ind = 1
  for el in data:
    str += '{}{}'.format(el, ', ')
    if ind % 5 == 0:
      str += '\n\t\t  '
    ind += 1
  return str[:-2]

st = data.groupby('id')['reviewId'].nunique().reset_index(name='count').sort_values(by='count')
st
#Составить общее мнение о фильме на основе отзывов

data

row, col = data.shape
print('Количество записей в наборе =', row)
print('Количество признаков в наборе =', col)

print('Список признаков:', indexes(data.columns))

data.info()

msno.matrix(data, color=(0, .6, .6))

data = data.drop('reviewUrl', axis=1)

duplicates = data[data.duplicated()]

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['reviewId'], label='Original Data')
plt.scatter(duplicates.index, duplicates['reviewId'], color='red', label='Duplicates')
plt.xlabel('Index')
plt.ylabel('Values in reviewId')
plt.title('Visualization of Duplicate Rows')
plt.legend()
plt.show()

data = data.drop_duplicates()

print('Количество критиков = %d' % data['criticName'].nunique())
print('Критики :', indexes(data['criticName'].head(50).unique()), 'and others...')

print('Количество фильмов = %d' % data['id'].nunique())
print('Фильмы :', indexes(data['id'].head(500).unique()), 'and others...')

critics = data.groupby('isTopCritic')['criticName'].nunique().reset_index(name='count')
critics_nan = data[(data['reviewText'].isna() == False)].groupby('isTopCritic')['criticName'].nunique().reset_index(name='count')
print(critics)
print(critics_nan)

reviews = data.groupby('reviewState')['reviewId'].count().reset_index(name='count')
reviews_nan = data[(data['reviewText'].isna() == False)].groupby('reviewState')['reviewId'].count().reset_index(name='count')
print(reviews)
print(reviews_nan)

sentiment = data.groupby('scoreSentiment')['reviewId'].count().reset_index(name='count')
sentiment_nan = data[(data['reviewText'].isna() == False)].groupby('scoreSentiment')['reviewId'].count().reset_index(name='count')
print(sentiment)
print(sentiment_nan)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
colors = sns.color_palette('GnBu', 2)

axes[0, 0].pie(critics['count'], labels=critics['isTopCritic'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[0, 0].set_title('The ratio of critics (all records)')

axes[0, 1].pie(critics_nan['count'], labels=critics_nan['isTopCritic'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[0, 1].set_title('The ratio of critics (without nan text)')

axes[1, 0].pie(reviews['count'], labels=reviews['reviewState'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[1, 0].set_title('The ratio of review state (fresh/rotten) (all records)')

axes[1, 1].pie(reviews_nan['count'], labels=reviews_nan['reviewState'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[1, 1].set_title('The ratio of review state (fresh/rotten) (without nan text)')

axes[2, 0].pie(sentiment['count'], labels=sentiment['scoreSentiment'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[2, 0].set_title('The ratio of score sentiment (NEGATIVE/POSITIVE) (all records)')

axes[2, 1].pie(sentiment_nan['count'], labels=sentiment_nan['scoreSentiment'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[2, 1].set_title('The ratio of score sentiment (NEGATIVE/POSITIVE) (without nan text)')

plt.tight_layout()
plt.show()

#x/10
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*10\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*10\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: x[0]).astype(float)
data.loc[(data['originalScore'].str.contains(r'^\d+(\.\d+)?/10-$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\d+(\.\d+)?/10-$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: x[0])
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*10\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\d+(\.\d+)?/10-$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan

#x/20
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*20\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*20\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.5)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*40\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*40\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.25)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*50\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*50\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.2)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*55\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*55\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.181)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*70\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*70\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.143)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*80\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*80\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.125)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*90\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*90\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.111)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*100\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*100\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.1)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*1000\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*1000\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.01)

data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*1000\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*100\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*90\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*80\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*70\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*55\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*50\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*40\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*20\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan

#x/5
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*9(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*9(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 1.111)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*8(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*8(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 1.25)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*7(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*7(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 1.429)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*6(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*6(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 1.666)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*5(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*5(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 2)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*4(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*4(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 2.5)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*3(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*3(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 3.333)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*2(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*2(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 5)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*1(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*1(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 10)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*0(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = 0

data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*13(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*13(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.769)
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*11(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*11(\.0)?\s*$')) & (data['originalScore'].isna() == False)]['originalScore'].str.split('/').apply(lambda x: np.float32(x[0]) * 0.909)

data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*13(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*11(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan

data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*9(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*8(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*7(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*6(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*5(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*4(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*3(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*2(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*1(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*(\d+)?(\.)?(\d+)?\s*/\s*0(\.0)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan

data.loc[((data['originalScore'].str.contains(r'^\s*[Aa]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Aa]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 10.000
data.loc[((data['originalScore'].str.contains(r'^\s*[Aa]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Aa]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 8.889
data.loc[((data['originalScore'].str.contains(r'^\s*[Bb]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Bb]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 8.333
data.loc[((data['originalScore'].str.contains(r'^\s*[Bb]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Bb]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 7.222
data.loc[((data['originalScore'].str.contains(r'^\s*[Cc]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Cc]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 6.667
data.loc[((data['originalScore'].str.contains(r'^\s*[Cc]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Cc]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 5.556
data.loc[((data['originalScore'].str.contains(r'^\s*[Dd]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Dd]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 5.000
data.loc[((data['originalScore'].str.contains(r'^\s*[Dd]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Dd]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 3.889
data.loc[((data['originalScore'].str.contains(r'^\s*[Ee]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Ee]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 3.333
data.loc[((data['originalScore'].str.contains(r'^\s*[Ee]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Ee]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 2.222
data.loc[((data['originalScore'].str.contains(r'^\s*[Ff]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Ff]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 3.333
data.loc[((data['originalScore'].str.contains(r'^\s*[Ff]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Ff]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'score'] = 0.000

data.loc[(data['originalScore'].str.contains(r'^\s*[Aa]\s*$')) & (data['originalScore'].isna() == False), 'score'] = 9.444
data.loc[(data['originalScore'].str.contains(r'^\s*[Bb]\s*$')) & (data['originalScore'].isna() == False), 'score'] = 7.778
data.loc[(data['originalScore'].str.contains(r'^\s*[Cc]\s*$')) & (data['originalScore'].isna() == False), 'score'] = 6.111
data.loc[(data['originalScore'].str.contains(r'^\s*[Dd]\s*$')) & (data['originalScore'].isna() == False), 'score'] = 4.444
data.loc[(data['originalScore'].str.contains(r'^\s*[Ee]\s*$')) & (data['originalScore'].isna() == False), 'score'] = 2.778
data.loc[(data['originalScore'].str.contains(r'^\s*[Ff]\s*$')) & (data['originalScore'].isna() == False), 'score'] = 1.111

data.loc[(data['originalScore'].str.contains(r'^\s*[Aa]\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*[Bb]\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*[Cc]\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*[Dd]\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*[Ee]\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[(data['originalScore'].str.contains(r'^\s*[Ff]\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan

data.loc[((data['originalScore'].str.contains(r'^\s*[Aa]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Aa]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Aa]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Aa]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Bb]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Bb]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Bb]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Bb]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Cc]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Cc]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Cc]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Cc]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Dd]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Dd]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Dd]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Dd]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Ee]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Ee]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Ee]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Ee]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Ff]\s*(\+)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Ff]\s*(\-)?\s*plus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan
data.loc[((data['originalScore'].str.contains(r'^\s*[Ff]\s*(\-)*\s*$')) | (data['originalScore'].str.contains(r'^\s*[Ff]\s*(\-)?\s*minus\s*$'))) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan

data.loc[(data['originalScore'].str.contains(r'^\s*\d+(\.\d+)?\s*$')) & (data['originalScore'].isna() == False), 'score'] = data[data['originalScore'].str.contains(r'^\s*\d+(\.\d+)?\s*$') & (data['originalScore'].isna() == False)]['originalScore']
data.loc[(data['originalScore'].str.contains(r'^\s*\d+(\.\d+)?\s*$')) & (data['originalScore'].isna() == False), 'originalScore'] = np.nan

data['originalScore'] = data['originalScore'].str.strip()

data['originalScore'] = data['originalScore'].str.replace('\'', '')
data['originalScore'] = data['originalScore'].str.replace('`', '')
data['originalScore'] = data['originalScore'].str.replace('=', '+')
data['originalScore'] = data['originalScore'].str.replace('_', '-')
data['originalScore'] = data['originalScore'].str.replace('?', '')
data['originalScore'] = data['originalScore'].str.replace('+-', '')
data['originalScore'] = data['originalScore'].str.replace('-+', '')
data['originalScore'] = data['originalScore'].str.replace('*', '')
data['originalScore'] = data['originalScore'].str.replace('%', '')
data['originalScore'] = data['originalScore'].str.replace('m inus', '-')
data['originalScore'] = data['originalScore'].str.replace('min us', '-')
data['originalScore'] = data['originalScore'].str.replace('- - -', '-')
data['originalScore'] = data['originalScore'].str.replace(' out of 10', '')
data['originalScore'] = data['originalScore'].str.replace(' of 10', '')
data['originalScore'] = data['originalScore'].str.replace('0 stars', '0')
data['originalScore'] = data['originalScore'].str.replace('-10', '0')
data['originalScore'] = data['originalScore'].str.replace('\\', '/')

data['originalScore'] = data['originalScore'].str.replace('ZERO STARS', '0')
data['originalScore'] = data['originalScore'].str.replace('ONE STAR', '2')
data['originalScore'] = data['originalScore'].str.replace('TWO STARS', '4')
data['originalScore'] = data['originalScore'].str.replace('THREE STARS', '6')
data['originalScore'] = data['originalScore'].str.replace('8 STARS', '8')
data['originalScore'] = data['originalScore'].str.replace('FOUR STARS', '8')
data['originalScore'] = data['originalScore'].str.replace('FIVE STARS', '10')

data.loc[(data['originalScore'] == '.5 stars') | (data['originalScore'] == '5 stars') | (data['originalScore'] == '5/5 stars'), 'score'] = '10'
data.loc[(data['originalScore'] == '4 1/2 stars') | (data['originalScore'] == '4 1/2 out of 5 stars'), 'score'] = '9'
data.loc[(data['originalScore'] == '4/5 stars') | (data['originalScore'] == '4 stars out of 5') | (data['originalScore'] == '4 out of 5 stars') | (data['originalScore'] == '4 stars'), 'score'] = '8'
data.loc[(data['originalScore'] == '3/4 stars'), 'score'] = '7.5'
data.loc[(data['originalScore'] == '3.5 out of 5 stars') | (data['originalScore'] == '3 1/2 stars') | (data['originalScore'] == '3.5 stars'), 'score'] = '7'
data.loc[(data['originalScore'] == '3/5 stars') | (data['originalScore'] == '3 stars'), 'score'] = '6'
data.loc[(data['originalScore'] == '2/4 stars') | (data['originalScore'] == '2.5 stars') | (data['originalScore'] == '2 1/2 stars') | (data['originalScore'] == '2.5/5 stars'), 'score'] = '5'
data.loc[(data['originalScore'] == 'two stars') | (data['originalScore'] == '2 stars out of 5') | (data['originalScore'] == '2 stars'), 'score'] = '4'
data.loc[(data['originalScore'] == '1.5 stars'), 'score'] = '3'
data.loc[(data['originalScore'] == '1/4 stars'), 'score'] = '2.5'
data.loc[(data['originalScore'] == '1 star') | (data['originalScore'] == '1-5 stars') | (data['originalScore'] == '1-5 Stars') | (data['originalScore'] == '1 out of 5 stars') | (data['originalScore'] == '1 stars'), 'score'] = '2'
data.loc[(data['originalScore'] == '1/2 star'), 'score'] = '1'


data.loc[(data['originalScore'] == '8.5/8.5'), 'score'] = '10'

data.loc[(data['originalScore'] == '8.5/8.5'), 'originalScore'] = np.nan

data.loc[(data['originalScore'] == '3.5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 out of 5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1-5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1-5 Stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4/5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2.5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1.5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1 out of 5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 stars out of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 1/2 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2 1/2 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1/2 star'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5 out of 5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3/5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 1/2 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1/4 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3/4 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2 stars out of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1 star'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2.5/5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2/4 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '5/5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'two stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '.5 stars'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 1/2 out of 5 stars'), 'originalScore'] = np.nan

data.loc[(data['originalScore'] == 'Recommended'), 'score'] = True
data.loc[(data['originalScore'] == 'short'), 'score'] = True
data.loc[(data['originalScore'] == 'Low Half Price'), 'score'] = True
data.loc[(data['originalScore'] == 'Highly Recommended'), 'score'] = True
data.loc[(data['originalScore'] == 'Catch It On Cable'), 'score'] = True
data.loc[(data['originalScore'] == 'Fresh'), 'score'] = True
data.loc[(data['originalScore'] == 'Big Screen Watch'), 'score'] = True
data.loc[(data['originalScore'] == 'bomb'), 'score'] = True
data.loc[(data['originalScore'] == 'Half Price'), 'score'] = True
data.loc[(data['originalScore'] == 'BOMB!'), 'score'] = True
data.loc[(data['originalScore'] == 'Bomb'), 'score'] = True
data.loc[(data['originalScore'] == 'Low Big Screen Watch'), 'score'] = True
data.loc[(data['originalScore'] == 'Schmoed it up'), 'score'] = True
data.loc[(data['originalScore'] == 'Sid Smith'), 'score'] = True
data.loc[(data['originalScore'] == 'G.O.A.T'), 'score'] = True
data.loc[(data['originalScore'] == 'Overripe'), 'score'] = True
data.loc[(data['originalScore'] == 'AWFUL!!!'), 'score'] = True
data.loc[(data['originalScore'] == 'Olivia Wilde loses the makeup that made her the fa'), 'score'] = True
data.loc[(data['originalScore'] == 'Matinee'), 'score'] = True
data.loc[(data['originalScore'] == 'Recomended'), 'score'] = True
data.loc[(data['originalScore'] == 'Turkey!'), 'score'] = True
data.loc[(data['originalScore'] == 'RECOMMENDED'), 'score'] = True
data.loc[(data['originalScore'] == 'matinee'), 'score'] = True
data.loc[(data['originalScore'] == 'GOAT'), 'score'] = True
data.loc[(data['originalScore'] == 'zb'), 'score'] = True
data.loc[(data['originalScore'] == 'Rental'), 'score'] = True
data.loc[(data['originalScore'] == 'High Rental'), 'score'] = True
data.loc[(data['originalScore'] == 'Better Than Sex'), 'score'] = True
data.loc[(data['originalScore'] == 'Muy Buena'), 'score'] = True
data.loc[(data['originalScore'] == 'BIts still fun, but you feel more like youre wat'), 'score'] = True
data.loc[(data['originalScore'] == 'Read A Book'), 'score'] = False
data.loc[(data['originalScore'] == 'aint great'), 'score'] = False
data.loc[(data['originalScore'] == 'Strongly Not Recommended'), 'score'] = False
data.loc[(data['originalScore'] == 'capsule'), 'score'] = False
data.loc[(data['originalScore'] == 'Not Recommended'), 'score'] = False

data.loc[(data['originalScore'] == 'Recommended'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'short'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Low Half Price'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Highly Recommended'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Catch It On Cable'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Fresh'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Big Screen Watch'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'bomb'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Half Price'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'BOMB!'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Bomb'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Low Big Screen Watch'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Schmoed it up'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Sid Smith'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'G.O.A.T'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Overripe'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'AWFUL!!!'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Olivia Wilde loses the makeup that made her the fa'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Matinee'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Recomended'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Turkey!'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'RECOMMENDED'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'matinee'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'GOAT'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'zb'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Rental'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'High Rental'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Better Than Sex'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Muy Buena'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'BIts still fun, but you feel more like youre wat'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Read A Book'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'aint great'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Strongly Not Recommended'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'capsule'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'Not Recommended'), 'originalScore'] = np.nan

data.loc[(data['originalScore'] == 'high +1 out of -4..+4'), 'score'] = '7'
data.loc[(data['originalScore'] == 'high 0 out of -4..+4'), 'score'] = '5'
data.loc[(data['originalScore'] == 'high +2 out of -4..+4'), 'score'] = '8'
data.loc[(data['originalScore'] == '3 out 5'), 'score'] = '6'
data.loc[(data['originalScore'] == '2 Ripples'), 'score'] = '4'
data.loc[(data['originalScore'] == '0 out of -4..+4'), 'score'] = '5'
data.loc[(data['originalScore'] == '+1 out of -4..+4'), 'score'] = '1'
data.loc[(data['originalScore'] == '4 of 5'), 'score'] = '8'
data.loc[(data['originalScore'] == '2 of 5'), 'score'] = '4'
data.loc[(data['originalScore'] == '3.5/18'), 'score'] = '1.944'
data.loc[(data['originalScore'] == '4..5/5'), 'score'] = '9'

data.loc[(data['originalScore'] == 'high +1 out of -4..+4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'high 0 out of -4..+4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'high +2 out of -4..+4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 out 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2 Ripples'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '0 out of -4..+4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '+1 out of -4..+4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2 of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5/18'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4..5/5'), 'originalScore'] = np.nan

data.loc[(data['originalScore'] == '2.5/5Too tepid and too predictable of a picture to') | (data['originalScore'] == '2.5/') | (data['originalScore'] == '2/4/') | (data['originalScore'] == '2.5 out of 5')| (data['originalScore'] == '2. 5 / 5') | (data['originalScore'] == '2.5 of 5') | (data['originalScore'] == '.5'), 'score'] = '5'
data.loc[(data['originalScore'] == '3/.5') | (data['originalScore'] == '3 out of 5 air tanks') | (data['originalScore'] == '3of 5') | (data['originalScore'] == '3 Ripples') | (data['originalScore'] == '60é100') | (data['originalScore'] == '3 of 5'), 'score'] = '6'
data.loc[(data['originalScore'] == '1/8/5') | (data['originalScore'] == '2/5/4'), 'score'] = '6.25'
data.loc[(data['originalScore'] == '7.5.10') | (data['originalScore'] == '3-4'), 'score'] = '7.5'
data.loc[(data['originalScore'] == '3/5.5') | (data['originalScore'] == '3/5/5') | (data['originalScore'] == '7 our'), 'score'] = '7'
data.loc[(data['originalScore'] == '4 Ripples'), 'score'] = '8'
data.loc[(data['originalScore'] == '4/3.5') | (data['originalScore'] == '3.5/44'), 'score'] = '8.75'
data.loc[(data['originalScore'] == '4 1/2/5'), 'score'] = '9'
data.loc[(data['originalScore'] == '5 of 5'), 'score'] = '10'
data.loc[(data['originalScore'] == '1 of 5') | (data['originalScore'] == '1-5'), 'score'] = '2'
data.loc[(data['originalScore'] == 'F (but also an A)'), 'score'] = '1.111'
data.loc[(data['originalScore'] == '2 1/2'), 'score'] = '2.5'
data.loc[(data['originalScore'] == '9/19'), 'score'] = '4.72'
data.loc[(data['originalScore'] == '3 1/2'), 'score'] = '3.5'
data.loc[(data['originalScore'] == '0 to 4'), 'score'] = '0'
data.loc[(data['originalScore'] == '3/35'), 'score'] = '0.86'
data.loc[(data['originalScore'] == '1/4ANNAPOL'), 'score'] = '2.5'
data.loc[(data['originalScore'] == '3 outta 5  aint great'), 'score'] = '6'
data.loc[(data['originalScore'] == '2.5.4'), 'score'] = '6.25'
data.loc[(data['originalScore'] == '3/4.'), 'score'] = '7.5'
data.loc[(data['originalScore'] == '3 of 4'), 'score'] = '7.5'
data.loc[(data['originalScore'] == '3 out of 5'), 'score'] = '6'
data.loc[(data['originalScore'] == 'B-/C+'), 'score'] = '6.667'
data.loc[(data['originalScore'] == '0 out of 4..+4'), 'score'] = '5'
data.loc[(data['originalScore'] == '2/5.4'), 'score'] = '6.25'
data.loc[(data['originalScore'] == '3.5/'), 'score'] = '7'
data.loc[(data['originalScore'] == '95/95'), 'score'] = '10'
data.loc[(data['originalScore'] == '4/2.5'), 'score'] = '6.25'
data.loc[(data['originalScore'] == '4/5.5'), 'score'] = '9'
data.loc[(data['originalScore'] == '3.5/5.5'), 'score'] = '6.36'
data.loc[(data['originalScore'] == '4 out of 5'), 'score'] = '8'
data.loc[(data['originalScore'] == '1.5/'), 'score'] = '3'
data.loc[(data['originalScore'] == '3.5 out of 5'), 'score'] = '7'
data.loc[(data['originalScore'] == '3.5/58'), 'score'] = '0.6'
data.loc[(data['originalScore'] == '21/2 pink ponies out of five'), 'score'] = '4'
data.loc[(data['originalScore'] == '3 out of 4 Ripples'), 'score'] = '7.5'
data.loc[(data['originalScore'] == '3 out of 4'), 'score'] = '7.5'
data.loc[(data['originalScore'] == '3. 5/5'), 'score'] = '7'
data.loc[(data['originalScore'] == '1/2/5'), 'score'] = '1'
data.loc[(data['originalScore'] == '-1/4'), 'score'] = '2.5'
data.loc[(data['originalScore'] == '2.5 Ripples'), 'score'] = '5'
data.loc[(data['originalScore'] == '3.5/51'), 'score'] = '0.68'
data.loc[(data['originalScore'] == '3/4f'), 'score'] = '7.5'
data.loc[(data['originalScore'] == '1.5 of 4'), 'score'] = '3.75'
data.loc[(data['originalScore'] == '3.5 of 4'), 'score'] = '8.75'
data.loc[(data['originalScore'] == '4/5http://www.bbc.co.uk/films/2007/08/27/a-throw-o'), 'score'] = '8'
data.loc[(data['originalScore'] == '0 out of 5'), 'score'] = '0'
data.loc[(data['originalScore'] == '61/00'), 'score'] = '6.1'
data.loc[(data['originalScore'] == 'w.5/4'), 'score'] = '6.25'
data.loc[(data['originalScore'] == 'B/B-'), 'score'] = '7.778'
data.loc[(data['originalScore'] == '3,5.5'), 'score'] = '7'
data.loc[(data['originalScore'] == '-1/10'), 'score'] = '1'
data.loc[(data['originalScore'] == '2/4The best that can be said about the movie is th'), 'score'] = '5'
data.loc[(data['originalScore'] == '1/5/4'), 'score'] = '3.75'
data.loc[(data['originalScore'] == '4.5 out of 5'), 'score'] = '9'
data.loc[(data['originalScore'] == '2.5 of 4') | (data['originalScore'] == '2.5./4'), 'score'] = '6.25'
data.loc[(data['originalScore'] == '5/53'), 'score'] = '10'
data.loc[(data['originalScore'] == '3.5.4.0') | (data['originalScore'] == '3.5./4') | (data['originalScore'] == '3.5/4A') | (data['originalScore'] == '3 1/2  out of 4') | (data['originalScore'] == '3/5.4') | (data['originalScore'] == '3/5/4'), 'score'] = '8.75'
data.loc[(data['originalScore'] == '0.5/5.5'), 'score'] = '0.9'
data.loc[(data['originalScore'] == '4/5w') | (data['originalScore'] == '4/5ao'), 'score'] = '8'
data.loc[(data['originalScore'] == '1/54'), 'score'] = '0.2'
data.loc[(data['originalScore'] == '3.0/.0') | (data['originalScore'] == '3  of 5'), 'score'] = '6'
data.loc[(data['originalScore'] == '4.5 out of 5.'), 'score'] = '9'
data.loc[(data['originalScore'] == '3.5 of 5'), 'score'] = '7'
data.loc[(data['originalScore'] == '21/2/4'), 'score'] = '5'
data.loc[(data['originalScore'] == '2/'), 'score'] = '4'
data.loc[(data['originalScore'] == '6,5.10'), 'score'] = '6.5'
data.loc[(data['originalScore'] == '3/45'), 'score'] = '0.67'
data.loc[(data['originalScore'] == '1.5/.5'), 'score'] = '3'


data.loc[(data['originalScore'] == ''), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2.5/5Too tepid and too predictable of a picture to') | (data['originalScore'] == '2.5/') | (data['originalScore'] == '2/4/') | (data['originalScore'] == '2.5 out of 5')| (data['originalScore'] == '2. 5 / 5') | (data['originalScore'] == '2.5 of 5') | (data['originalScore'] == '.5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3/.5') | (data['originalScore'] == '3 out of 5 air tanks') | (data['originalScore'] == '3of 5') | (data['originalScore'] == '3 Ripples') | (data['originalScore'] == '60é100') | (data['originalScore'] == '3 of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1/8/5') | (data['originalScore'] == '2/5/4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '7.5.10') | (data['originalScore'] == '3-4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3/5.5') | (data['originalScore'] == '3/5/5') | (data['originalScore'] == '7 our'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 Ripples'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4/3.5') | (data['originalScore'] == '3.5/44'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 1/2/5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '5 of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1 of 5') | (data['originalScore'] == '1-5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'F (but also an A)'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2 1/2'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '9/19'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 1/2'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '0 to 4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3/35'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1/4ANNAPOL'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 outta 5  aint great'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2.5.4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3/4.'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 of 4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 out of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'B-/C+'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '0 out of 4..+4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2/5.4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5/'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '95/95'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4/2.5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4/5.5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5/5.5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4 out of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1.5/'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5 out of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5/58'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '21/2 pink ponies out of five'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 out of 4 Ripples'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3 out of 4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3. 5/5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1/2/5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '-1/4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2.5 Ripples'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5/51'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3/4f'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1.5 of 4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5 of 4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4/5http://www.bbc.co.uk/films/2007/08/27/a-throw-o'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '0 out of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '61/00'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'w.5/4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == 'B/B-'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3,5.5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '-1/10'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2/4The best that can be said about the movie is th'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1/5/4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4.5 out of 5'), 'originalScore'] = '9'
data.loc[(data['originalScore'] == '2.5 of 4') | (data['originalScore'] == '2.5./4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '5/53'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5.4.0') | (data['originalScore'] == '3.5./4') | (data['originalScore'] == '3.5/4A') | (data['originalScore'] == '3 1/2  out of 4') | (data['originalScore'] == '3/5.4') | (data['originalScore'] == '3/5/4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '0.5/5.5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4/5w') | (data['originalScore'] == '4/5ao'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1/54'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.0/.0') | (data['originalScore'] == '3  of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '4.5 out of 5.'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3.5 of 5'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '21/2/4'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '2/'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '6,5.10'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '3/45'), 'originalScore'] = np.nan
data.loc[(data['originalScore'] == '1.5/.5'), 'originalScore'] = np.nan

print('Количество оценок = %d' % data['originalScore'].nunique())
print('Оценки :', indexes(data['originalScore'].unique()))

data['score'] = data['score'].astype(float)
data.loc[(data['score'] > 10), 'score'] = data[data['score'] > 10]['score'] / 10
data[data['score'] > 10]

data = data.drop(data[data['reviewText'].isna()].index)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

df = data[['id', 'reviewId', 'criticName', 'isTopCritic', 'reviewState',	'score', 'reviewText', 'scoreSentiment']]
df

def clean(text):
  text = str(text)
  text = ''.join([i for i in text if not i.isdigit()])
  words = word_tokenize(text.lower())
  filter = [word for word in words if word not in stop_words and word not in punctuation]
  return " ".join(filter)

df['reviewText'] = df['reviewText'].str.lower()
df['reviewText'] = df['reviewText'].apply(clean)
df

text_tokens = df['reviewText'].apply(lambda x:  word_tokenize(x))
text_tokens

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')



def sentiment(text):
  sid = SentimentIntensityAnalyzer()
  scores = sid.polarity_scores(text)
  return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

df.loc[data.head(100000).index, ['neg', 'neu', 'pos', 'compound']] = pd.DataFrame(df['reviewText'].head(100000).apply(lambda x: sentiment(x)).tolist(), columns=['neg', 'neu', 'pos', 'compound'], index=df.head(100000).index)

df.head(10000)

cdata = data[['id', 'reviewId', 'criticName', 'isTopCritic', 'reviewText', 'score', 'reviewState',	'scoreSentiment']].drop(data[data['score'].isna()].index)
cdata['criticName'] = cdata['criticName'].str.strip()
cdata

row, col = cdata.shape
print('Количество записей в наборе =', row)
print('Количество признаков в наборе =', col)

print('Список признаков:', indexes(cdata.columns))

cdata.info()

print('Количество критиков = %d' % cdata['criticName'].nunique())
print('Критики :', indexes(cdata['criticName'].head(50).unique()), 'and others...')

print('Количество фильмов = %d' % cdata['id'].nunique())
print('Фильмы :', indexes(cdata['id'].head(500).unique()), 'and others...')

critics = cdata.groupby('isTopCritic')['criticName'].nunique().reset_index(name='count')
reviews = cdata.groupby('reviewState')['reviewId'].count().reset_index(name='count')
sentiment = cdata.groupby('scoreSentiment')['reviewId'].count().reset_index(name='count')
print(critics, '\n')
print(reviews, '\n')
print(sentiment)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
colors = sns.color_palette('GnBu', 2)

axes[0].pie(critics['count'], labels=critics['isTopCritic'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[0].set_title('The ratio of critics')

axes[1].pie(reviews['count'], labels=reviews['reviewState'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[1].set_title('The ratio of review state (fresh/rotten)')

axes[2].pie(sentiment['count'], labels=sentiment['scoreSentiment'], colors=colors, startangle=30, autopct='%1.3f%%')
axes[2].set_title('The ratio of score sentiment (NEGATIVE/PISITIVE)')

plt.tight_layout()
plt.show()

critics = cdata.groupby(['criticName', 'isTopCritic'])['reviewText'].nunique().reset_index(name='count').sort_values(by='count', ascending=False)
t_critics = cdata[cdata['isTopCritic'] == True].groupby(['criticName', 'isTopCritic'])['reviewText'].nunique().reset_index(name='count').sort_values(by='count', ascending=False)
nt_critics = cdata[cdata['isTopCritic'] == False].groupby(['criticName', 'isTopCritic'])['reviewText'].nunique().reset_index(name='count').sort_values(by='count', ascending=False)
print(critics, '\n')
print(t_critics, '\n')
print(nt_critics)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 45))

labels = critics['criticName'].head(50)
width = critics['count'].head(50)

colors = ['coral' if value == True else 'mediumturquoise' for value in critics['isTopCritic']]

axes[0].barh(labels, width, color=colors)
for index, value in enumerate(critics['count'].head(50)):
    axes[0].annotate(value, xy=(value+15, index-0.07), color='dimgray')
axes[0].set_xlabel('Count')
axes[0].set_title('The number of reviews by critic (all)')

labels = t_critics['criticName'].head(50)
width = t_critics['count'].head(50)

axes[1].barh(labels, width, color='mediumturquoise')
for index, value in enumerate(t_critics['count'].head(50)):
    axes[1].annotate(value, xy=(value+15, index-0.07), color='dimgray')
axes[1].set_xlabel('Count')
axes[1].set_title('The number of reviews by critic (top=true)')

labels = nt_critics['criticName'].head(50)
width = nt_critics['count'].head(50)

axes[2].barh(labels, width, color='mediumturquoise')
for index, value in enumerate(nt_critics['count'].head(50)):
    axes[2].annotate(value, xy=(value+15, index-0.07), color='dimgray')
axes[2].set_xlabel('Count')
axes[2].set_title('The number of reviews by critic (top=false)')

plt.tight_layout()
plt.show()

critics = cdata.groupby(['criticName', 'reviewState']).size().unstack(fill_value=0)
critics = critics.reset_index()
critics['total'] = critics[['fresh', 'rotten']].sum(axis=1)
critics = critics.sort_values(by='total')
print(critics)

fig, ax = plt.subplots(figsize=(10, 10))

ax.bar(critics['criticName'].tail(50), critics['fresh'].tail(50), label='Positive Reviews', color='mediumturquoise')
ax.bar(critics['criticName'].tail(50), critics['rotten'].tail(50), bottom=critics['fresh'].tail(50), label='Negative Reviews', color='coral')

ax.set_xlabel('Critic')
ax.set_ylabel('Number of Reviews')
ax.set_title('Positive and Negative Reviews per Critic')
ax.legend()

plt.xticks(rotation=90)

plt.show()

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

short = cdata.head(100000)

def sentiment(text):
  sid = SentimentIntensityAnalyzer()
  scores = sid.polarity_scores(text)
  return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

short.loc[short.index, ['neg', 'neu', 'pos', 'compound']] = pd.DataFrame(short['reviewText'].apply(lambda x: sentiment(x)).tolist(), columns=['neg', 'neu', 'pos', 'compound'], index=short.index)
short

short.loc[(short['compound'] >= 0), ['predicted']] = 'POSITIVE'
short.loc[(short['compound'] < 0), ['predicted']] = 'NEGATIVE'
short

colors = {'POSITIVE': 'mediumturquoise', 'NEGATIVE': 'coral'}
markers = {
    True: 'o',
    False: 'x'
}

plt.figure(figsize=(10, 6))

for i in range(len(short)):
    actual = short.iloc[i]['scoreSentiment']
    predicted = short.iloc[i]['predicted']
    is_match = actual == predicted
    plt.scatter(short.iloc[i]['compound'], short.iloc[i]['pos'],
                color=colors[actual],
                marker=markers[is_match],
                label=f"{actual}, {'match' if is_match else 'mismatch'}" if i == 0 else "")

plt.xlabel('Compound Score')
plt.ylabel('Positive Score')
plt.title('Sentiment Analysis: Prediction vs Actual')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend()
plt.show()

obs = short['reviewId'].count()
neq = short[short['scoreSentiment'] != short['predicted']]['reviewId'].count()
print('Количество записей = %d' % obs)
print('Количество несовпадений = %d' % neq)
print('Процент несовпадений = %.2f%%' % (neq / obs * 100))
