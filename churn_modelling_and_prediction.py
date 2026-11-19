import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
import missingno as msno
import warnings
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import classification_report
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

warnings.filterwarnings('ignore')

data = pd.read_csv('Gym churn us.csv')

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

row, col = data.shape
print('Количество записей в наборе =', row)
print('Количество признаков в наборе =', col)

print('Список признаков:', indexes(data.columns))

data = data.rename(columns={'gender' : 'gender',
                    'Near_Location' : 'isNear',
                    'Partner' : 'isPartner',
                    'Promo_friends' : 'isFriend',
                    'Phone' : 'isPhone',
                    'Contract_period' : 'contract',
                    'Group_visits' : 'isGroup',
                    'Age' : 'age',
                    'Avg_additional_charges_total' : 'avg_add_cost',
                    'Month_to_end_contract' : 'endContract',
                    'Lifetime' : 'lifetime',
                    'Avg_class_frequency_total' : 'avg_frequency',
                    'Avg_class_frequency_current_month' : 'avg_frequency_cur',
                    'Churn' : 'isChurn'})

data = data.reindex(columns=['gender', 'age', 'lifetime', 'isPhone', 'isNear',
                             'isPartner', 'isFriend', 'isGroup', 'contract',
                             'endContract', 'avg_frequency_cur', 'avg_frequency', 'avg_add_cost', 'isChurn'])

data = data.reindex(columns=['gender', 'age', 'lifetime', 'isPhone', 'isNear',
                             'isPartner', 'isFriend', 'isGroup', 'contract',
                             'endContract', 'avg_frequency_cur', 'avg_frequency', 'avg_add_cost', 'isLow', 'isLess', 'isChurn'])

data.info()

duplicates = data[data.duplicated()]

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['age'], label='Original Data', color='mediumturquoise')
plt.scatter(duplicates.index, duplicates['age'], color='coral', label='Duplicates')
plt.title('Visualization of duplicate rows')
plt.xlabel('Number of observation')
plt.ylabel('Age')
plt.legend()
plt.show()

data.describe()

by_gender = data.groupby(['gender']).size().reset_index(name='count')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))

axes.barh(by_gender['gender'].apply(lambda x : str(x)), by_gender['count'], height=0.7, color='mediumturquoise')
for index, value in enumerate(by_gender['count']):
    axes.annotate(value, xy=(value+15, index-0.07), color='slategray')

axes.set_title('Customers grouped by gender')
axes.set_ylabel('Quantity')
axes.set_xlabel('Gender')
axes.set_ylim(-1, 2)
axes.set_xlim(0, 2500)

plt.tight_layout()
plt.show()

print('Количество возрастов = %d' % data['age'].nunique())
print('Минимальный возраст клиента - %d лет' % min(data['age']))
print('Максимальный возраст клиента - %d год' % max(data['age']))

by_age = data.groupby(['age']).size().reset_index(name='count')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

axes.bar(by_age['age'], by_age['count'], color='mediumturquoise')
for index, value in enumerate(by_age['count']):
    axes.annotate(value, xy=(index+17.75, value+5), color='slategray')

axes.set_title('Customers grouped by age')
axes.set_ylabel('Quantity')
axes.set_xlabel('Age')
axes.set_ylim(0, 550)

plt.tight_layout()
plt.show()

isPhone = data.groupby(['isPhone']).size().reset_index(name='count')
isNear = data.groupby(['isNear']).size().reset_index(name='count')
isPartner = data.groupby(['isPartner']).size().reset_index(name='count')
isFriend = data.groupby(['isFriend']).size().reset_index(name='count')
isGroup = data.groupby(['isGroup']).size().reset_index(name='count')
isChurn = data.groupby(['isChurn']).size().reset_index(name='count')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))

axes[0, 0].barh(isPhone['isPhone'].apply(lambda x : str(x)), isPhone['count'], height=0.7, color='mediumturquoise')
axes[0, 1].barh(isNear['isNear'].apply(lambda x : str(x)), isNear['count'], height=0.7, color='mediumturquoise')
axes[1, 0].barh(isGroup['isGroup'].apply(lambda x : str(x)), isGroup['count'], height=0.7, color='mediumturquoise')
axes[1, 1].barh(isPartner['isPartner'].apply(lambda x : str(x)), isPartner['count'], height=0.7, color='mediumturquoise')
axes[2, 0].barh(isFriend['isFriend'].apply(lambda x : str(x)), isFriend['count'], height=0.7, color='mediumturquoise')
axes[2, 1].barh(isChurn['isChurn'].apply(lambda x : str(x)), isChurn['count'], height=0.7, color='mediumturquoise')

for index, value in enumerate(isPhone['count']):
    axes[0, 0].annotate(value, xy=(value+20, index-0.07), color='slategray')

for index, value in enumerate(isNear['count']):
    axes[0, 1].annotate(value, xy=(value+20, index-0.07), color='slategray')

for index, value in enumerate(isGroup['count']):
    axes[1, 0].annotate(value, xy=(value+20, index-0.07), color='slategray')

for index, value in enumerate(isPartner['count']):
    axes[1, 1].annotate(value, xy=(value+20, index-0.07), color='slategray')

for index, value in enumerate(isFriend['count']):
    axes[2, 0].annotate(value, xy=(value+20, index-0.07), color='slategray')

for index, value in enumerate(isChurn['count']):
    axes[2, 1].annotate(value, xy=(value+20, index-0.07), color='slategray')

axes[0, 0].set_title('Customers grouped by "isPhone"')
axes[0, 1].set_title('Customers grouped by "isNear"')
axes[1, 0].set_title('Customers grouped by "isGroup"')
axes[1, 1].set_title('Customers grouped by "isPartner"')
axes[2, 0].set_title('Customers grouped by "isFriend"')
axes[2, 1].set_title('Customers grouped by "isChurn"')

for ax in axes.flat:
  ax.set_xlabel('Quantity')
  ax.set_xlim(0, 4000)
  ax.set_ylim(-1, 2)

plt.tight_layout()
plt.show()

by_lifetime = data.groupby(['lifetime']).size().reset_index(name='count')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

axes.bar(by_lifetime['lifetime'].apply(lambda x : str(x)), by_lifetime['count'], color='mediumturquoise')
for index, value in enumerate(by_lifetime['count']):
    axes.annotate(value, xy=(index-0.35, value+10), color='slategray')

axes.set_title('Customers grouped by lifetime')
axes.set_ylabel('Quantity')
axes.set_xlabel('Lifetime (month)')
axes.set_ylim(0, 950)

plt.tight_layout()
plt.show()

by_contract = data.groupby(['contract']).size().reset_index(name='count')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

axes.barh(by_contract['contract'].apply(lambda x : str(x)), by_contract['count'], height=0.7, color='mediumturquoise')
for index, value in enumerate(by_contract['count']):
    axes.annotate(value, xy=(value+20, index-0.07), color='slategray')

axes.set_title('Customers grouped by contract')
axes.set_xlabel('Quantity')
axes.set_ylim(-1, 3)
axes.set_xlim(0, 2500)

plt.tight_layout()
plt.show()

data[[ 'isLow', 'isLess', 'isChurn']].corr().style.background_gradient(cmap = 'GnBu')

data.corr().style.background_gradient(cmap = 'GnBu')

gender_vs_churn = data.pivot_table(index='gender', columns='isChurn', values='avg_add_cost', aggfunc='count', fill_value=0).reset_index()
isPhone_vs_churn = data.pivot_table(index='isPhone', columns='isChurn', values='avg_add_cost', aggfunc='count', fill_value=0).reset_index()
isNear_vs_churn = data.pivot_table(index='isNear', columns='isChurn', values='avg_add_cost', aggfunc='count', fill_value=0).reset_index()
isLow_vs_churn = data.pivot_table(index='isLow', columns='isChurn', values='avg_add_cost', aggfunc='count', fill_value=0).reset_index()
gender_vs_churn.columns.name = None
isPhone_vs_churn.columns.name = None
isNear_vs_churn.columns.name = None
isLow_vs_churn.columns.name = None

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

sns.heatmap(gender_vs_churn.set_index('gender'), annot=True, cmap='GnBu', fmt='.0f', ax=axes[0, 0], cbar_kws={'label': 'Count'})
sns.heatmap(isPhone_vs_churn.set_index('isPhone'), annot=True, cmap='GnBu', fmt='.0f', ax=axes[0, 1], cbar_kws={'label': 'Count'})
sns.heatmap(isNear_vs_churn.set_index('isNear'), annot=True, cmap='GnBu', fmt='.0f', ax=axes[1, 0], cbar_kws={'label': 'Count'})
sns.heatmap(isLow_vs_churn.set_index('isLow'), annot=True, cmap='GnBu', fmt='.0f', ax=axes[1, 1], cbar_kws={'label': 'Count'})

axes[0, 0].set_ylabel('Gender')
axes[0, 1].set_ylabel('isPhone')
axes[1, 0].set_ylabel('isNear')
axes[1, 1].set_ylabel('isLow')

for ax in axes.flat:
  ax.set_xlabel('isChurn')

age_vs_churn = data.pivot_table(index='age', columns='isChurn', values='avg_add_cost', aggfunc='count', fill_value=0).reset_index()
age_vs_churn.columns.name = None

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

sns.heatmap(age_vs_churn.set_index('age'), annot=True, cmap='GnBu', fmt='.0f', ax=axes, cbar_kws={'label': 'Count'})

axes.set_ylabel('Age')
axes.set_xlabel('isChurn')

lg_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

def evaluate(model, x, y):
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    return accuracy, precision, recall, f1, roc_auc

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['isChurn'], test_size=0.2, random_state=42)

lg_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
gb_model.fit(x_train, y_train)

lg_accuracy, lg_precision, lg_recall, lg_f1, lg_roc_auc = evaluate(lg_model, x_test, y_test)
rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate(rf_model, x_test, y_test)
gb_accuracy, gb_precision, gb_recall, gb_f1, gb_roc_auc = evaluate(gb_model, x_test, y_test)

print('Результат оценки моделей:')
print('\n___Логистическая регрессия')
print(f'\tТочность = {lg_accuracy:.4f}\n\tПолнота = {lg_recall:.4f}\n\tF1-мера = {lg_f1:.4f}\n\tROC-AUC = {lg_roc_auc:.4f}')
print('\n___Случайный лес')
print(f'\tТочность = {rf_accuracy:.4f}\n\tПолнота = {rf_recall:.4f}\n\tF1-мера = {rf_f1:.4f}\n\tROC-AUC = {rf_roc_auc:.4f}')
print('\n___Градиентный бустинг')
print(f'\tТочность = {gb_accuracy:.4f}\n\tПолнота = {gb_recall:.4f}\n\tF1-мера = {gb_f1:.4f}\n\tROC-AUC = {gb_roc_auc:.4f}')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

y_pred = gb_model.predict(x_test)

axes.scatter(x_test.index, y_test, color='mediumturquoise', label='Actual')
axes.scatter(x_test.index, y_pred, color='coral', label='Predicted')
axes.set_title('Forecasting customer churn')
axes.set_xlabel('Observation')
axes.set_ylabel('Is churn')
axes.set_ylim(-1, 2)
axes.legend()

plt.tight_layout()
plt.show()

feature_importances = pd.DataFrame({'Feature': data.iloc[:, :-1].columns, 'Importance': gb_model.feature_importances_})
print(feature_importances.sort_values(by='Importance', ascending=False))
print()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

axes.barh(feature_importances['Feature'], feature_importances['Importance'], color='mediumturquoise', align='center')
for index, value in enumerate(feature_importances['Importance']):
    axes.annotate(format('%.3f' % value), xy=(value+0.002, index-0.15), color='slategray')

axes.set_title('Feature Importances')
axes.set_xlabel('Relative Importance')
axes.set_ylabel('Features')
axes.set_xlim(0, 0.5)

plt.show()

new_obs = pd.DataFrame({
    'gender': [1],
    'age': [20],
    'lifetime': [5],
    'isPhone': [0],
    'isNear': [0],
    'isPartner': [0],
    'isFriend': [0],
    'isGroup': [0],
    'contract': [1],
    'endContract': [0],
    'avg_frequency_cur': [0],
    'avg_frequency': [0],
    'avg_add_cost': [100],
})

new_pred = gb_model.predict(new_obs)

print('\nПредсказанные значения для новых наблюдений = %s' % indexes(new_pred))


# Пример использования модели для расчета вероятности
def predict_churn_probability(avg_frequency, avg_frequency_cur):
  new_obs = pd.DataFrame({
    'gender': [1],
    'age': [20],
    'lifetime': [5],
    'isPhone': [0],
    'isNear': [0],
    'isPartner': [0],
    'isFriend': [0],
    'isGroup': [0],
    'contract': [1],
    'endContract': [0],
    'avg_frequency_cur': [avg_frequency],
    'avg_frequency': [avg_frequency_cur],
    'avg_add_cost': [100],
})

  return gb_model.predict_proba(new_obs)[0, 1]

example_avg_frequency = 2
example_avg_frequency_cur = 0.1
probability = predict_churn_probability(example_avg_frequency, example_avg_frequency_cur)
print(f"Вероятность оттока: {probability:.2f}")

clusters = linkage(data, method='ward')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

dendrogram(clusters, labels=data.index.tolist(), ax=axes, color_threshold=0.7 * max(clusters[:, 1]), above_threshold_color='mediumturquoise')

axes.set_title('Dendrogram')
axes.set_xlabel('Observation')
axes.set_ylabel('Distance')

plt.show()

scaler = StandardScaler()

features = ['age', 'lifetime', 'avg_frequency_cur', 'avg_frequency', 'avg_add_cost']
scaled = scaler.fit_transform(data[features])

kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled)

data['churnProbability'] = gb_model.predict_proba(data.iloc[:, :13])[:, 1]

cluster_analysis = data.groupby('cluster').agg({
    'gender' : lambda x: x.mode()[0],
    'age' : 'mean',
    'lifetime' : 'mean',
    'isPhone' : lambda x: x.mode()[0],
    'isNear' : lambda x: x.mode()[0],
    'isPartner' : lambda x: x.mode()[0],
    'isFriend' : lambda x: x.mode()[0],
    'isGroup' : lambda x: x.mode()[0],
    'contract' : lambda x: x.mode()[0],
    'endContract' : 'mean',
    'avg_frequency_cur' : 'mean',
    'avg_frequency' : 'mean',
    'avg_add_cost' : 'mean',
    #'churnProbability' : ['mean', 'median']
})

cluster_analysis

cluster_analysis = data.groupby('cluster')['churnProbability'].agg(['mean', 'median'])

cluster_analysis

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

x = data[data['cluster'] == 2].iloc[:, :13]
y = data[data['cluster'] == 2]['isChurn'].to_numpy()

y_pred = gb_model.predict(x)

axes.scatter(x.index, y, color='mediumturquoise', label='Actual')
axes.scatter(x.index, y_pred, color='coral', label='Predicted')
axes.set_title('Forecasting customer churn')
axes.set_xlabel('Observation')
axes.set_ylabel('Is churn')
axes.set_ylim(-1, 2)
axes.legend()

plt.tight_layout()
plt.show()

data['avg_frequency_cur'].mean()

data['isLow'] = 0
data.loc[(data['avg_frequency_cur'] < data['avg_frequency_cur'].mean()), 'isLow'] = 1

data['isLess'] = 0
data.loc[(data['avg_frequency_cur'] < data['avg_frequency']), 'isLess'] = 1
