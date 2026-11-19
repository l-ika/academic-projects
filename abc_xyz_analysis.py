import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
import missingno as msno
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

data = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/E-commercial dataset.xlsx')

def indexes(data):
  str = ''
  ind = 1
  for el in data:
    str += '{}{}'.format(el, ', ')
    if ind % 7 == 0:
      str += '\n\t   '
    ind += 1
  return str[:-2]

data

row, col = data.shape
print('Количество записей в наборе =', row)
print('Количество признаков в наборе =', col)

print('Список признаков:', indexes(data.columns))

"""<h1>Поля в датасете</h1>
<ul>
  <li><strong>ID</strong> &#0151; идентификатор каждого клиента</li>
  <li><strong>Birth</strong> &#0151; год рождения</li>
  <li><strong>Education</strong> &#0151; уровень образования</li>
  <li><strong>MaritalStatus</strong> &#0151; семейное положение</li>
  <li><strong>Income</strong> &#0151; доход &#0151; деньги, полученные в результате выполнения работы или инвестиций</li>
  <li><strong>Registration</strong> &#0151; дата регистрации клиента в компании</li>
  <li><strong>Recency</strong> &#0151; время, прошедшее с посленей покупки</li>
  <li><strong>mnt_Clothes</strong> &#0151; сумма, потраченная на одежду</li>
  <li><strong>mnt_Electronics</strong> &#0151; сумма, потраченная на технику и электронику</li>
  <li><strong>mnt_Books</strong> &#0151; сумма, потраченная на книги</li>
  <li><strong>mnt_Furniture</strong> &#0151; сумма, потраченная на мебель</li>
  <li><strong>mnt_Foods</strong> &#0151; сумма, потраченная на продукты питания</li>
  <li><strong>mnt_Cosmetics</strong> &#0151; сумма, потраченная на косметику</li>
  <li><strong>num_Deals</strong> &#0151; количество сделок, которые клиент совершал каждый раз при совершении покупок</li>
  <li><strong>num_Web</strong> &#0151; количество веб-сайтов, которые клиент использовал для совершения покупок в Интернете</li>
  <li><strong>num_WebVisitsMonth</strong> &#0151; количество веб-сайтов, которые клиент посещал ежемесячно</li>
  <li><strong>PaymentMethod</strong> &#0151; способ оплаты, который клиент использовал при совершении покупок в Интернете</li>
  <li><strong>isDiscount</strong> &#0151; была ли применена скидка к клиенту (True &#0151; да, False &#0151; нет)</li>
  <li><strong>isComplain</strong> &#0151; оставил ли клиент какие-либо жалобы</li>
  <li><strong>isResponse</strong> &#0151; оставил ли клиент какие-либо ответы</li>
  <li><strong>Satisfaction</strong> &#0151; оценка степени удовлетворенности клиента после получения продукции</li>
  <li><strong>avg_Rating</strong> &#0151; как клиент оценивает продукцию</li>
  <li><strong>isChurns</strong> &#0151; был ли совершен отказ (0 &#0151; нет, 1 &#0151; да)</li>
</ul>

"""

data = data.drop('Unnamed: 0', axis=1)

data = data.rename(columns={'ID' : 'id',
                    'Year_Birth' : 'birth',
                    'Education' : 'education',
                    'Marital_Status' : 'maritalStatus',
                    'Income' : 'income',
                    'Dt_Customer' : 'registration',
                    'Recency' : 'recency',
                    'MntClothes' : 'mnt_Clothes',
                    'MntTech&Electronics' : 'mnt_Electronics',
                    'MntBooks' : 'mnt_Books',
                    'MntFurniture' : 'mnt_Furniture',
                    'MntFoods' : 'mnt_Foods',
                    'MntCosmetics' : 'mnt_Cosmetics',
                    'NumDealsPurchases' : 'num_Deals',
                    'NumWebPurchases' : 'num_Web',
                    'NumWebVisitsMonth' : 'num_WebVisitsMonth',
                    'Payment_Method' : 'paymentMethod',
                    'Discount_Applied' : 'isDiscount',
                    'Complain' : 'isComplain',
                    'Response' : 'isResponse',
                    'Satisfaction_Level' : 'satisfaction',
                    'Average_Rating' : 'avg_Rating',
                    'Churns' : 'isChurn'})

data.info()

msno.matrix(data, color=(0, 0.6, 0.6))

data[(data['income'].isna() | data['satisfaction'].isna())]

avg_income = data['income'].mean()
median_income = data['income'].median()

print('Среднее значение \'income\' = %.2f' % avg_income)
print('Медианное значение \'income\' = %.2f' % median_income)

data.loc[(data['income'].isna()), 'income'] = median_income
data.loc[(data['satisfaction'].isna()), 'satisfaction'] = 'Neutral'

duplicates = data[data.duplicated()]

plt.figure(figsize=(15, 7))
plt.scatter(data.index, data['income'], label='Original Data', color='mediumturquoise')
plt.scatter(duplicates.index, duplicates['income'], color='coral', label='Duplicates')
plt.title('Visualization of duplicate rows')
plt.xlabel('Number of observation')
plt.ylabel('Income')
plt.legend()
plt.show()

print('Количество уникальных значений поля \'birth\' = %d' % data['birth'].nunique())
print('Значения :', indexes(data['birth'].sort_values().unique()), '\n')

print('Количество уникальных значений поля \'education\' = %d' % data['education'].nunique())
print('Значения :', indexes(data['education'].sort_values().unique()), '\n')

print('Количество уникальных значений поля \'maritalStatus\' = %d' % data['maritalStatus'].nunique())
print('Значения :', indexes(data['maritalStatus'].sort_values().unique()), '\n')

print('Количество уникальных значений поля \'paymentMethod\' = %d' % data['paymentMethod'].nunique())
print('Значения :', indexes(data['paymentMethod'].sort_values().unique()), '\n')

print('Количество уникальных значений поля \'satisfaction\' = %d' % data['satisfaction'].nunique())
print('Значения :', indexes(data['satisfaction'].sort_values().unique()), '\n')

display(data[data['birth'] < 1920])
data = data.drop(data[data['birth'] < 1920].index)

display(data[(data['maritalStatus'] == 'YOLO') | (data['maritalStatus'] == 'Absurd') | (data['maritalStatus'] == 'Alone')])
data = data.drop(data[(data['maritalStatus'] == 'YOLO') | (data['maritalStatus'] == 'Absurd')].index)

data.loc[(data['maritalStatus'] == 'Alone'), 'maritalStatus'] = 'Single'
data.loc[(data['isDiscount'] == True), 'isDiscount'] = 1
data.loc[(data['isDiscount'] == False), 'isDiscount'] = 0

"""<h1>Визуализация данных</h1>"""

full_range = pd.DataFrame({'birth': range(min(data['birth']), max(data['birth']) + 1)})
by_birth = pd.merge(full_range, data.groupby(['birth']).size().reset_index(name='count'), on='birth', how='left').fillna(0)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

axes.bar(by_birth['birth'], by_birth['count'], color='mediumturquoise')
for index, value in enumerate(by_birth['count']):
    axes.annotate(format(value, '.0f'), xy=(index+min(data['birth']), value+2), color='slategray', ha='center')

axes.set_title('Customers grouped by birth')
axes.set_ylabel('Quantity')
axes.set_xlabel('Year of birth')
axes.set_ylim(0, 110)

plt.tight_layout()
plt.show()

by_education = data.groupby(['education']).size().reset_index(name='count').sort_values(by='count')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
colors = sns.color_palette('GnBu', len(by_education['education']))

axes[0].pie(by_education['count'], labels=by_education['education'], colors=colors, startangle=10, autopct='%1.1f%%')
axes[0].set_title('By education')

axes[1].barh(by_education['education'], by_education['count'], color='mediumturquoise')
for index, value in enumerate(by_education['count']):
    axes[1].annotate(value, xy=(value+10, index), color='slategray', va='center')

axes[1].set_title('Customers grouped by education')
axes[1].set_ylabel('Education')
axes[1].set_xlabel('Quantity')
axes[1].set_xlim(0, 1300)

plt.tight_layout()
plt.show()

by_marital = data.groupby(['maritalStatus']).size().reset_index(name='count').sort_values(by='count')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
colors = sns.color_palette('GnBu', len(by_marital['maritalStatus']))

axes[0].pie(by_marital['count'], labels=by_marital['maritalStatus'], colors=colors, startangle=10, autopct='%1.1f%%')
axes[0].set_title('By marital status')

axes[1].barh(by_marital['maritalStatus'], by_marital['count'], color='mediumturquoise')
for index, value in enumerate(by_marital['count']):
    axes[1].annotate(value, xy=(value+10, index), color='slategray', va='center')

axes[1].set_title('Customers grouped by marital status')
axes[1].set_ylabel('Marital status')
axes[1].set_xlabel('Quantity')
axes[1].set_xlim(0, 1300)

plt.tight_layout()
plt.show()

by_payment = data.groupby(['paymentMethod']).size().reset_index(name='count').sort_values(by='count')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
colors = sns.color_palette('GnBu', len(by_payment['paymentMethod']))

axes[0].pie(by_payment['count'], labels=by_payment['paymentMethod'], colors=colors, startangle=10, autopct='%1.1f%%')
axes[0].set_title('By payment method')

axes[1].barh(by_payment['paymentMethod'], by_payment['count'], color='mediumturquoise')
for index, value in enumerate(by_payment['count']):
    axes[1].annotate(value, xy=(value+10, index), color='slategray', va='center')

axes[1].set_title('Customers grouped by payment method')
axes[1].set_ylabel('Payment method')
axes[1].set_xlabel('Quantity')
axes[1].set_xlim(0, 1000)
axes[1].set_ylim(-1, 3)

plt.tight_layout()
plt.show()

by_satisfaction = data.groupby(['satisfaction']).size().reset_index(name='count').sort_values(by='count')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
colors = sns.color_palette('GnBu', len(by_satisfaction['satisfaction']))

axes[0].pie(by_satisfaction['count'], labels=by_satisfaction['satisfaction'], colors=colors, startangle=10, autopct='%1.1f%%')
axes[0].set_title('By satisfaction level')

axes[1].barh(by_satisfaction['satisfaction'], by_satisfaction['count'], color='mediumturquoise')
for index, value in enumerate(by_satisfaction['count']):
    axes[1].annotate(value, xy=(value+10, index), color='slategray', va='center')

axes[1].set_title('Customers grouped by satisfaction level')
axes[1].set_ylabel('Satisfaction level')
axes[1].set_xlabel('Quantity')
axes[1].set_xlim(0, 1000)
axes[1].set_ylim(-1, 3)

plt.tight_layout()
plt.show()

isDiscount = data.groupby(['isDiscount']).size().reset_index(name='count')
isComplain = data.groupby(['isComplain']).size().reset_index(name='count')
isResponse = data.groupby(['isResponse']).size().reset_index(name='count')
isChurn = data.groupby(['isChurn']).size().reset_index(name='count')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

axes[0, 0].barh(isDiscount['isDiscount'].apply(lambda x : str(x)), isDiscount['count'], height=0.7, color='mediumturquoise')
axes[0, 1].barh(isComplain['isComplain'].apply(lambda x : str(x)), isComplain['count'], height=0.7, color='mediumturquoise')
axes[1, 0].barh(isResponse['isResponse'].apply(lambda x : str(x)), isResponse['count'], height=0.7, color='mediumturquoise')
axes[1, 1].barh(isChurn['isChurn'].apply(lambda x : str(x)), isChurn['count'], height=0.7, color='mediumturquoise')

for index, value in enumerate(isDiscount['count']):
    axes[0, 0].annotate(value, xy=(value+20, index-0.07), color='slategray')

for index, value in enumerate(isComplain['count']):
    axes[0, 1].annotate(value, xy=(value+20, index-0.07), color='slategray')

for index, value in enumerate(isResponse['count']):
    axes[1, 0].annotate(value, xy=(value+20, index-0.07), color='slategray')

for index, value in enumerate(isChurn['count']):
    axes[1, 1].annotate(value, xy=(value+20, index-0.07), color='slategray')

axes[0, 0].set_title('Customers grouped by "isDiscount"')
axes[0, 1].set_title('Customers grouped by "isComplain"')
axes[1, 0].set_title('Customers grouped by "isResponse"')
axes[1, 1].set_title('Customers grouped by "isChurn"')

for ax in axes.flat:
  ax.set_xlabel('Quantity')
  ax.set_xlim(0, 2500)
  ax.set_ylim(-1, 2)

plt.tight_layout()
plt.show()

isDiscount_vs_churn = data.pivot_table(index='isDiscount', columns='isChurn', values='income', aggfunc='count', fill_value=0).reset_index()
isDiscount_vs_churn.columns.name = None

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

sns.heatmap(isDiscount_vs_churn.set_index('isDiscount'), annot=True, cmap='GnBu', fmt='.0f', ax=axes, cbar_kws={'label': 'Count'})

axes.set_ylabel('isDiscount')
axes.set_xlabel('isChurn')

plt.tight_layout()
plt.show()

label_encoder = LabelEncoder()
excl = [5]

dcorr = data.iloc[:, [i for i in range(0, 23, 1) if i not in excl]]
dcorr['education'] = label_encoder.fit_transform(dcorr['education'])
dcorr['maritalStatus'] = label_encoder.fit_transform(dcorr['maritalStatus'])
dcorr['paymentMethod'] = label_encoder.fit_transform(dcorr['paymentMethod'])

dcorr.loc[(dcorr['satisfaction'] == 'Satisfied'), 'satisfaction'] = 2
dcorr.loc[(dcorr['satisfaction'] == 'Neutral'), 'satisfaction'] = 1
dcorr.loc[(dcorr['satisfaction'] == 'Unsatisfied'), 'satisfaction'] = 0

dcorr.corr().style.background_gradient(cmap='GnBu')

"""<h1>ABC/XYZ анализ</h1>"""

abc = data[['id']]
abc['amount'] = data[['mnt_Clothes', 'mnt_Electronics', 'mnt_Books', 'mnt_Furniture', 'mnt_Foods', 'mnt_Cosmetics']].sum(axis=1)
abc = abc.sort_values(by='amount', ascending=False)

total_amount = abc['amount'].sum()
abc['amount_share'] = abc['amount'] / total_amount * 100
abc['total_amount_share'] = abc['amount_share'].cumsum()
abc.loc[(abc['total_amount_share'] <= 80), 'group'] = 'A'
abc.loc[((abc['total_amount_share'] > 80) & (abc['total_amount_share'] <= 95)), 'group'] = 'B'
abc.loc[(abc['total_amount_share'] > 95), 'group'] = 'C'
abc

count_A = abc[abc['group'] == 'A'].shape[0]
count_B = abc[abc['group'] == 'B'].shape[0]
count_C = abc[abc['group'] == 'C'].shape[0]
print('Группа клиентов А - лидеры = %d (%.2f%%)' % (count_A, count_A / abc.shape[0] * 100))
print('Группа клиентов B - середнячки = %d (%.2f%%)' % (count_B, count_B / abc.shape[0] * 100))
print('Группа клиентов C - аутсайдеры = %d (%.2f%%)' % (count_C, count_C / abc.shape[0] * 100))

by_abc = abc.groupby(['group']).size().reset_index(name='count').sort_values(by='count')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
colors = sns.color_palette('GnBu', len(by_abc['group']))

axes.pie(by_abc['count'], labels=by_abc['group'], colors=colors, startangle=10, autopct='%1.1f%%')
axes.set_title('By ABC group')

plt.tight_layout()
plt.show()

def variation(row):
    return (row.std() / row.mean()) * 100

xyz = data[['id', 'mnt_Clothes', 'mnt_Electronics', 'mnt_Books', 'mnt_Furniture', 'mnt_Foods', 'mnt_Cosmetics']]
xyz['variation'] = xyz.iloc[:, 1:].apply(variation, axis=1)
xyz = xyz.sort_values(by='variation')

xyz.loc[(xyz['variation'] <= 50), 'group'] = 'X'
xyz.loc[((xyz['variation'] > 50) & (xyz['variation'] <= 75)), 'group'] = 'Y'
xyz.loc[(xyz['variation'] > 75), 'group'] = 'Z'
xyz

count_X = xyz[xyz['group'] == 'X'].shape[0]
count_Y = xyz[xyz['group'] == 'Y'].shape[0]
count_Z = xyz[xyz['group'] == 'Z'].shape[0]
print('Группа клиентов X - постоянные покупатели = %d (%.2f%%)' % (count_X, count_X / xyz.shape[0] * 100))
print('Группа клиентов Y - целевые покупатели = %d (%.2f%%)' % (count_Y, count_Y / xyz.shape[0] * 100))
print('Группа клиентов Z - редкие покупатели = %d (%.2f%%)' % (count_Z, count_Z / xyz.shape[0] * 100))

by_xyz = xyz.groupby(['group']).size().reset_index(name='count').sort_values(by='count')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
colors = sns.color_palette('GnBu', len(by_xyz['group']))

axes.pie(by_xyz['count'], labels=by_xyz['group'], colors=colors, startangle=10, autopct='%1.1f%%')
axes.set_title('By XYZ group')

plt.tight_layout()
plt.show()

summary = pd.merge(abc[['id', 'group']], xyz[['id', 'group']], on='id')
data = pd.merge(summary, data, on='id')

pivot = summary.pivot_table(index='group_y', columns='group_x', values='id', aggfunc='count', fill_value=0).reset_index()
pivot

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

sns.heatmap(pivot.set_index('group_y'), annot=True, cmap='GnBu_r', fmt='.0f', ax=axes, cbar_kws={'label': 'Count'})

axes.set_ylabel('Group Y')
axes.set_xlabel('Group X')

data.loc[summary[(summary['group_x'] == 'A') & (summary['group_y'] == 'X')].index]

"""<h2><strong>Портрет клиента</strong></h2>"""

portrait_ax = data.groupby(['group_x', 'group_y']).agg({
    'birth' : lambda x : x.mode()[0],
    'education' : lambda x : x.mode()[0],
    'maritalStatus' : lambda x : x.mode()[0],
    'income' : 'mean',
    'recency' : 'mean',
    'mnt_Clothes' : 'median',
    'mnt_Electronics' : 'median',
    'mnt_Books' : 'median',
    'mnt_Furniture' : 'median',
    'mnt_Foods' : 'median',
    'mnt_Cosmetics' : 'median',
    'num_Deals' : 'mean',
    'num_Web' : 'mean',
    'num_WebVisitsMonth' : 'mean',
    'paymentMethod' : lambda x : x.mode()[0],
    'avg_Rating' : 'mean',

})

portrait_ax

"""<h1><strong>Генерация текстов</strong></h1>"""

!pip install -U transformers

import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it"
)

"""<h2><font color='mediumturquoise'>Для группы AX</font></h2>
<ul>
  <li><font size='2'>A &#0151; высокие суммарные траты</font></li>
  <li><font size='2'>X &#0151; низкая вариация трат (постоянные покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают значительные траты и их покупки стабильны во всех категориях. Они постоянные и лояльные клиенты, на которых можно рассчитывать в долгосрочной перспективе. Им можно предлагать эксклюзивные предложения и программы лояльности для поддержания их интереса.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make significant expenses and their purchases are stable in all categories. They are regular and loyal customers who can be counted on in the long run. They can be offered exclusive offers and loyalty programs to maintain their interest.'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who spend heavily and their purchases are stable in all categories. These are regular customers who can be counted on in the long run. To maintain interest in them, you can offer exclusive offers and loyalty programs.'
            }]

output_1 = pipe(message_1, max_new_tokens=256)
output_2 = pipe(message_2, max_new_tokens=256)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  Elevate Your Productivity with the Xiaomi Redmi Book 14

**Dear valued customer,**

We know you appreciate quality, performance, and innovation. You're not just a buyer, you're a discerning individual who values long-lasting value and exceptional experiences.

That's why we're excited to introduce the **Xiaomi Redmi Book 14**, a laptop designed to meet the demands of your demanding lifestyle.

**Here's what sets the Redmi Book 14 apart:**

* **Uncompromising Performance:** Powered by the latest [Processor Name], the Redmi Book 14 delivers lightning-fast processing speeds and seamless multitasking, whether you're editing high-resolution images, running demanding software, or streaming your favorite content.
* **Stunning Visuals:** Immerse yourself in vibrant colors and crystal-clear details with the stunning [Display Size] display. Experience a truly cinematic viewing experience with [Display Resolution] resolution and [Display Technology].
* **Unmatched Battery Life:**  Work or play for hours on end without worrying about power. The Redmi Book 14 boasts a long-lasting battery that keeps you connected and productive throughout the day.

##  You're More Than Just a Customer, You're a Partner.

We know you've built a successful business, and we're proud to be a part of your journey.  You're a consistent, reliable partner, and we value your trust and loyalty.

**That's why we're offering you something special:**

* **Exclusive Access:**  Enjoy early access to new products, special discounts, and invitations to exclusive events.
* **Personalized Rewards:**  Earn points on every purchase that can be redeemed for exclusive benefits, like personalized discounts, free shipping, or even merchandise.
* **Dedicated Support:**  Our team is here to provide you with personalized support and guidance, ensuring your needs are met every step of the way.

**We believe in building long-term relationships, and we want to make sure you feel valued and appreciated.**  

**Ready to unlock exclusive benefits and elevate your experience?**

**Contact us today to learn more about our loyalty program and how we can tailor our services to your specific needs.**

**[Contact Information]**

##  Upgrade Your Workday with the Power of Xiaomi! ⚡️

**Hey [Customer Name],**

Remember that sleek, powerful laptop you've been eyeing?  Well, now's your chance to experience the future of productivity with the **Xiaomi Laptop!**

**Here's why you'll love it:**

* **Unleash your creativity:**  Enjoy a vibrant, high-resolution display perfect for work, school, or creative projects. 🎨
* **Lightning-fast performance:**  Experience seamless multitasking and lag-free gaming with the powerful processor and ample RAM. 🎮
* **Long-lasting battery:**  Power through your day without worrying about charging. 🔋
* **Sleek and stylish design:**  Make a statement with a laptop that's as beautiful as it is functional. ✨
* **Exclusive discounts for our valued customers:**  Get [mention specific discount, e.g., 10% off] on your Xiaomi Laptop purchase! 🎁

**Don't miss out on this incredible opportunity to upgrade your tech game!**

**Visit us today at [Store Address] or shop online at [Website Link] and use code [Discount Code] at checkout.**

**

<h2><font color='mediumturquoise'>Для группы AY</font></h2>
<ul>
  <li><font size='2'>A &#0151; высокие суммарные траты</font></li>
  <li><font size='2'>Y &#0151; средняя вариация трат (целевые покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают значительные траты, но их предпочтения могут меняться. Они активно покупают товары в определенных категориях, и им можно предлагать акции на популярные у них категории товаров.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make significant expenses but their preferences may change. They actively buy products in certain categories, and they can be offered promotions for popular product categories.'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who make significant expenses but their preferences may change. They actively buy products in certain categories, and they can be offered promotions for popular product categories.'
            }]

output_1 = pipe(message_1, max_new_tokens=256)
output_2 = pipe(message_2, max_new_tokens=256)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  Hey [Customer Name], We've Got Something Special Just For You!

We know you value quality and convenience, and we're always working to make your shopping experience even better.  

**We've noticed you're a frequent buyer in [mention specific categories, e.g., electronics, home appliances, travel gear].**  We're excited to offer you exclusive promotions on products you love, and maybe even introduce you to some new favorites!

**Here's how we can make your shopping even more rewarding:**

* **Personalized Recommendations:** We'll send you tailored product suggestions based on your past purchases and browsing history.
* **Early Access to Sales:** Be the first to know about exclusive deals and limited-time offers on your favorite categories.
* **Exclusive Discounts:** Enjoy special promotions and rewards just for you, based on your purchase history.

**Ready to explore new possibilities and save even more?**

[Link to your website or specific promotions page]

**We're here to help you find the perfect products for your needs, and we're confident you'll love what we have to offer.**

##  Tired of Compromising on Performance?

**Upgrade your work experience with the Xiaomi Redmi Book 14.**  

You know the drill: you're a discerning buyer, always seeking the best in quality and performance. You've got your eye on the latest tech, and you're not afraid to invest in what you need.

**But what if you could have it all?**  

The Redmi Book 14 delivers a powerful performance without breaking the bank.  

**Here's what you'll love:**

* **Unleash your productivity:**  Experience lightning-fast processing with the latest Intel processor and ample RAM.
* **Stunning visuals:**  Immerse yourself in vibrant colors and crystal-clear displays with a high-resolution screen.
* **Long-lasting battery:**  Power through your day without worrying about charging.
* **Sleek and stylish design:**  Make a statement with a premium look and feel.

**And the best part?**  We know you value a good deal.  

**Right now, you can enjoy exclusive discounts on the Redmi Book 14.**  This is your chance to upgrade your work experience without sacrificing your budget.

<h2><font color='mediumturquoise'>Для группы AZ</font></h2>
<ul>
  <li><font size='2'>A &#0151; высокие суммарные траты</font></li>
  <li><font size='2'>Z &#0151; высокая вариация трат (нерегулярные покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают значительные траты, но их покупки нерегулярны и не предсказуемы. Их можно привлекать мотивационными скидками и специальными акциями, чтобы увеличить частоту покупок и сделать их более стабильными клиентами.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make significant expenditures, but their purchases are irregular and unpredictable. They can be attracted with motivational discounts and special promotions to increase the frequency of purchases and make them more stable customers.'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who make significant expenditures, but their purchases are irregular and unpredictable. They can be attracted with motivational discounts and special promotions to increase the frequency of purchases and make them more stable customers.'
            }]

output_1 = pipe(message_1, max_new_tokens=256)
output_2 = pipe(message_2, max_new_tokens=256)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  Unlock the Power of Predictable Purchases

**Are you a business that experiences significant, yet unpredictable spending patterns?**

We understand. You're a high-value customer who makes strategic investments, but the timing and frequency of those purchases can be difficult to predict.

**Here's how we can help:**

* **Unlock Predictable Purchases:**  We offer exclusive, **motivational discounts and special promotions** designed to encourage consistent, predictable spending.
* **Maximize Your ROI:**  By fostering a more regular purchasing cycle, you can optimize your budget and maximize the return on your investment.
* **Build Stronger Relationships:**  We're committed to building long-term partnerships with our valued customers. Our tailored programs ensure you receive the support and resources you need to succeed.

**Ready to take control of your spending and unlock the full potential of your business?**

**Contact us today to learn more about our exclusive programs and how we can help you achieve your goals.**

**[Call to Action: Contact us for a personalized consultation]**

**[Website Link]**

##  Unlock Your Productivity Potential:  Xiaomi Redmi Book 14, Now More Accessible Than Ever.

**You're a discerning buyer. You value quality, performance, and innovation.** You understand the power of a reliable device to elevate your work and unlock your potential.  

But you also know that big purchases aren't always easy to make.  Life throws curveballs, and your budget can be unpredictable.

**That's where Xiaomi Redmi Book 14 comes in.**

This powerhouse laptop delivers a premium experience at a price that won't break the bank.  With its sleek design, powerful processor, and long-lasting battery, the Redmi Book 14 is your perfect companion for work, creativity, and everything in between.

**But here's the best part:** We understand your needs.  We want you to experience the power of the Redmi Book 14, and we're offering special promotions to make it even more accessible.

**Here's what you can expect:**

* **Limited-time discounts:**  Get a significant discount on the Redmi Book 14, making it a truly affordable investment.
* **Flexible payment options:**  Choose a payment plan that fits your budget

<h2><font color='mediumturquoise'>Для группы BX</font></h2>
<ul>
  <li><font size='2'>B &#0151; средние суммарные траты</font></li>
  <li><font size='2'>X &#0151; низкая вариация трат (постоянные покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают средние суммарные траты и их покупки стабильны. Они часто покупают различные товары, но их траты не столь значительны, как у сегмента A. Им можно предлагать регулярные скидки и акции, чтобы стимулировать их к увеличению трат.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make average total spending and their purchases are stable. They often buy various goods, but their spending is not as significant as that of the A segment. They can be offered regular discounts and promotions to encourage them to increase spending. The text should be written according to the ODC (Offer + Deadline + Call to action) formula'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who make average total spending and their purchases are stable. They often buy various goods, but their spending is not as significant as that of the A segment. They can be offered regular discounts and promotions to encourage them to increase spending. The text should be written according to the ODC (Offer + Deadline + Call to action) formula'
            }]

output_1 = pipe(message_1, max_new_tokens=300)
output_2 = pipe(message_2, max_new_tokens=300)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  Upgrade Your Workday with the Xiaomi Redmi Book 14 -  Limited Time Offer!

**Offer:**  Get the powerful Xiaomi Redmi Book 14 at a special price! Enjoy a **15% discount** on the already affordable price, making it the perfect laptop for your everyday needs.

**Deadline:** This offer is valid for the next **7 days only**! Don't miss out on this chance to upgrade your work experience.

**Call to action:**  Visit our website or your nearest retailer to grab your Redmi Book 14 today!  

**Why choose the Redmi Book 14?**

* **Powerful Performance:**  Get everything done with ease thanks to its fast processor and ample RAM.
* **Stunning Display:** Enjoy vibrant colors and sharp details on the large, high-resolution screen.
* **Long-lasting Battery:** Work or play for hours on a single charge.
* **Affordable Price:**  Enjoy premium features without breaking the bank.

**Don't wait! This limited-time offer won't last forever. Upgrade your work experience with the Xiaomi Redmi Book 14 today!**

## Unlock More Savings with [Your Brand]!

Hey [Customer Name],

We know you're a valued customer who appreciates quality and variety. We appreciate your consistent purchases and want to help you get even more value out of your shopping experience.

That's why we're offering you a special 15% discount on your next order! This is a great opportunity to stock up on your favorite items or try something new.

Don't miss out on this limited-time offer! Shop now and enjoy the savings.

**[Link to your website or online store]**

See you soon!

The [Your Brand] Team

## Subject:  Unlock More Value with [Your Brand]

**Hi [Customer Name],**

We've noticed you're a regular shopper at [Your Brand], and we appreciate your continued support! We understand you enjoy browsing our wide selection of products, and we want to help you get even more value out of your shopping experience.

**Here's how we can help:**

* **Exclusive Discounts:** We're offering you a special discount on your next purchase! Use code **[Discount Code]** at checkout to save [Discount Percentage] off your total bill.
* **Personalized Recommendations:** We're always looking for ways to make your shopping experience even better. Tell us what you're looking for, and we'll suggest products you might love.
* **Loyalty Rewards:** Join our loyalty program and earn points on every purchase. Redeem your points for exclusive discounts and special offers.

**We believe in making your shopping experience enjoyable and rewarding.** By taking advantage of these offers, you can unlock even more value from [Your Brand].

**Shop Now:** [Link to your website]

**Sincerely,**

The [Your Brand] Team

##  Upgrade Your Everyday with the Xiaomi Redmi Book 14

**Tired of your laptop feeling slow and clunky?**  The Xiaomi Redmi Book 14 is here to change the game.  

**This powerful laptop delivers:**

* **Smooth performance:**  Get things done quickly and efficiently with the latest Intel processor and ample RAM.
* **Stunning visuals:** Enjoy vibrant colors and crisp details on the bright, high-resolution display.
* **Long-lasting battery:** Power through your day without worrying about finding an outlet.
* **All-day comfort:**  The comfortable keyboard and sleek design make it perfect for work or play.

**Plus, right now you can save even more with our special offer!**

**Don't miss out on this chance to upgrade your everyday experience.**  

**Click here to learn more and claim your discount!**

**[Link to product page]**

<h2><font color='mediumturquoise'>Для группы BY</font></h2>
<ul>
  <li><font size='2'>B &#0151; средние суммарные траты</font></li>
  <li><font size='2'>Y &#0151; средняя вариация трат (целевые покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают средние суммарные траты, и их предпочтения могут меняться. Они часто покупают товары в определенных категориях, поэтому акции на популярные категории помогут увеличить их траты.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make average total expenses, and their preferences may vary. They often buy products in certain categories, so promotions for popular categories will help increase their spending.'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who make average total expenses, and their preferences may vary. They often buy products in certain categories, so promotions for popular categories will help increase their spending.'
            }]

output_1 = pipe(message_1, max_new_tokens=300)
output_2 = pipe(message_2, max_new_tokens=300)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  Hey there, [Customer Name]!

We know you're busy juggling life's little (and big) moments.  That's why we're here to help you save time and money on the things you love.

**We've got something special for you today!**  

We're running a promotion on your favorite categories, like [mention popular categories like home goods, groceries, electronics, etc.].  Think [mention specific benefits like "saving on your weekly grocery haul", "getting the latest tech gadget at a great price", "making your home feel cozy with new bedding", etc.].

**Here's what you can expect:**

* **[Mention specific discount or promotion]** on [mention specific categories].
* **[Mention any other benefits like free shipping, bonus items, etc.]**
* **[Mention any limited-time offer]** so you don't miss out!

**Ready to treat yourself?**  

Click here to browse our amazing deals: [link to your website]

**Don't wait! This offer won't last forever.**

Happy shopping!

**The [Your Company Name] Team**

##  Work Smarter, Not Harder:  The Xiaomi Redmi Book 14 is Here!

**Tired of clunky laptops that drain your battery and slow you down?**  The **Xiaomi Redmi Book 14** is here to change the game.  

**This sleek and powerful laptop is perfect for:**

* **Students:**  Ace your classes with a lightweight, portable device that won't break the bank.
* **Professionals:**  Get things done efficiently with a fast processor, ample storage, and long battery life.
* **Anyone who values value:**  The Redmi Book 14 delivers top-notch performance without the premium price tag.

**Here's what makes it stand out:**

* **Stunning Display:**  Enjoy vibrant colors and crisp details on the 14-inch FHD display.
* **Lightning-Fast Performance:**  The powerful processor and ample RAM ensure smooth multitasking and effortless browsing.
* **Long-Lasting Battery:**  Work or play for hours on a single charge.
* **Sleek and Stylish Design:**  The Redmi Book 14 is as beautiful as it is functional.

**Plus, right now you can get it at an incredible price!**  

**Don't miss out on this limited-time offer!**  Upgrade your work experience and enjoy the power of the Xiaomi Redmi Book 14.

**Click here to learn more**

<h2><font color='mediumturquoise'>Для группы BZ</font></h2>
<ul>
  <li><font size='2'>B &#0151; средние суммарные траты</font></li>
  <li><font size='2'>Z &#0151; высокая вариация трат (нерегулярные покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают средние суммарные траты, о их покупки нерегулярны. Их можно привлекать специальными предложениями и скидками, чтобы увеличить частоту покупок и сделать их более стабильными клиентами.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make average total expenses, about their purchases are irregular. They can be attracted with special offers and discounts to increase the frequency of purchases and make them more stable customers.'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who make average total expenses, about their purchases are irregular. They can be attracted with special offers and discounts to increase the frequency of purchases and make them more stable customers.'
            }]

output_1 = pipe(message_1, max_new_tokens=300)
output_2 = pipe(message_2, max_new_tokens=300)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  👋  Hey there, [Customer Name]!  👋

We know you're busy, and your shopping habits are a bit... irregular.  We get it!  

But we're here to help you make your life easier and your shopping experience more rewarding.  

**Here's the deal:** We're offering you a special **[Discount Percentage]% off your next purchase** when you sign up for our loyalty program!  

**Here's what you get:**

* **Exclusive deals and discounts:**  Get access to special offers and promotions you won't find anywhere else.
* **Early access to new products:** Be the first to know about our latest arrivals and get exclusive previews.
* **Personalized recommendations:** We'll help you find the perfect products based on your past purchases and preferences.
* **Faster checkout:** Enjoy a seamless and hassle-free shopping experience.

**Think of it as a little something to help you stay organized and save money.**  

**Ready to join the [Your Brand Name] family?**  Click here to sign up: [Link to Loyalty Program Sign-Up]

**We're confident you'll love the benefits!**

**Sincerely,**

The [Your Brand Name] Team


##  Unlock Productivity & Savings with the Xiaomi Redmi Book 14!

**Tired of juggling work and life?**  The Xiaomi Redmi Book 14 is here to help you stay productive and save money.

**This powerful laptop is perfect for:**

* **Students:**  Effortlessly tackle assignments, research, and projects with its sleek design and fast performance.
* **Professionals:**  Streamline your workflow with its intuitive interface and powerful processor.
* **Anyone who needs a reliable workhorse:**  The Redmi Book 14 is built to last, with a durable chassis and long-lasting battery.

**But here's the best part:**  We're offering **exclusive deals** to help you make the most of your budget.

**Here's what you can expect:**

* **Special discounts:**  Save big on the Redmi Book 14 with our limited-time offers.
* **Flexible payment options:**  Choose a payment plan that fits your budget and schedule.
* **Free accessories:**  Get a free carrying case or other valuable accessories with your purchase.

**Don't miss out on this opportunity to upgrade your productivity and save money.**  

**Visit our website or your nearest retailer today and experience the Xiaomi Redmi Book 14 difference!**

**[Link to website or store]**

**Limited time offer!**

<h2><font color='mediumturquoise'>Для группы CX</font></h2>
<ul>
  <li><font size='2'>C &#0151; низкие суммарные траты</font></li>
  <li><font size='2'>X &#0151; низкая вариация трат (постоянные покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают низкие суммарные траты, но их покупки стабильны. Они не тратят много, но покупают регулярно. Им можно предлагать небольшие скидки и специальные предложения, чтобы стимулировать их к увеличению трат.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make low total expenses, but their purchases are stable. They don\'t spend much, but they buy regularly. They can be offered small discounts and special offers to encourage them to increase spending.'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who make low total expenses, but their purchases are stable. They don\'t spend much, but they buy regularly. They can be offered small discounts and special offers to encourage them to increase spending.'
            }]

output_1 = pipe(message_1, max_new_tokens=300)
output_2 = pipe(message_2, max_new_tokens=300)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  Hey [Customer Name], we see you!

We know you're a steady hand, always there for the essentials. You're a reliable customer, and we appreciate your consistent business.

**But we also know you're always looking for the best value.**  

That's why we're offering you a special opportunity to **unlock even more savings** on the products you love.

**Here's what we're thinking:**

* **[Mention a specific discount or offer, e.g., 5% off your next purchase]**
* **[Mention a special promotion, e.g., Free shipping on orders over [amount]]**
* **[Highlight a loyalty program or points system]**

We believe in building long-term relationships, and we want to help you get the most out of your purchases.

**Ready to explore the possibilities?**

[Link to your website or a specific product page]

We're here to help you find the perfect solutions for your needs.


**P.S.**  Don't forget to check out our [mention a new product or category] - we think you'll love it!



##  Need a Reliable Workhorse Without Breaking the Bank?

**Introducing the Xiaomi Redmi Book 14!**

You know the drill: you need a laptop, but your budget is tight. You're not looking for flashy features, just a solid machine that gets the job done.  

**The Redmi Book 14 is here to help.**

This powerful laptop packs a punch with:

* **Long-lasting battery:**  Work all day without worrying about finding an outlet.
* **Fast performance:**  Handle your daily tasks with ease, from browsing to writing.
* **Affordable price:**  Get top-notch performance without emptying your wallet.

**And here's the best part:** We're offering a special discount just for you!  

**Upgrade your work experience without overspending.  Get your Redmi Book 14 today!**

**Limited time offer!**  

**[Link to product page]**

<h2><font color='mediumturquoise'>Для группы CY</font></h2>
<ul>
  <li><font size='2'>C &#0151; низкие суммарные траты</font></li>
  <li><font size='2'>Y &#0151; средняя вариация трат (целевые покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают низкие суммарные траты, и их предпочтения могут меняться. Они покупают товары в определенных категориях, поэтому акции на популярные категории помогут увеличить их траты.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make low total expenses, and their preferences may vary. They buy products in certain categories, so promotions for popular categories will help increase their spending.'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who make low total expenses, and their preferences may vary. They buy products in certain categories, so promotions for popular categories will help increase their spending.'
            }]

output_1 = pipe(message_1, max_new_tokens=300)
output_2 = pipe(message_2, max_new_tokens=300)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  Unlock Savings & Discover New Favorites!

**Hey there, savvy shopper!**  We know you value your money, and we're here to help you stretch it further.

**You're a smart buyer who:**

* **Prioritizes value:** You're not just looking for the cheapest option, you want quality and reliability.
* **Loves variety:** You enjoy exploring new products and brands, but you stick to certain categories.
* **Values deals:** You're always on the lookout for promotions and discounts.

**That's why we've got something special for you!**

We're running **exclusive promotions** on your favorite categories, including:

* **[Category 1]:**  Get [Discount percentage] off your favorite [Product type]
* **[Category 2]:**  Stock up on essentials with [Discount percentage] off your entire purchase
* **[Category 3]:**  Discover new favorites with [Discount percentage] off select items

**Plus, we're offering:**

* **Free shipping** on orders over [Amount]
* **Exclusive loyalty rewards** for repeat customers
* **Easy returns** for your peace of mind

**Don't miss out on these incredible deals!**  Shop now and discover the perfect products for your needs and budget.

**[Link to your website]**




##  Work Smarter, Not Harder:  Xiaomi Redmi Book 14 for Budget-Conscious Professionals

**Tired of sacrificing performance for affordability?**  The Xiaomi Redmi Book 14 is here to change the game.  

**This powerful laptop packs a punch without breaking the bank.**  Get everything you need to stay productive, all while saving money.

**Here's why the Redmi Book 14 is perfect for you:**

* **Unbeatable Value:**  Enjoy a sleek, lightweight design and a powerful processor at a price that won't strain your budget.
* **Long-Lasting Battery:**  Work all day without worrying about finding an outlet. The Redmi Book 14's battery keeps you going, even on the go.
* **Essential Features:**  Get everything you need for work and beyond, including a spacious display, fast storage, and reliable connectivity.
* **Perfect for Everyday Tasks:**  From emails and spreadsheets to video calls and light photo editing, the Redmi Book 14 handles it all with ease.

**Special Offer:**  For a limited time, enjoy an exclusive discount on the Redmi Book 14.  This is the perfect opportunity to upgrade your work experience without overspending.

**Don't settle for less. Choose the Redmi Book 14 and experience the power of productivity without the price tag.**

**Click here to learn more and claim your discount!**

<h2><font color='mediumturquoise'>Для группы CZ</font></h2>
<ul>
  <li><font size='2'>C &#0151; низкие суммарные траты</font></li>
  <li><font size='2'>Z &#0151; высокая вариация трат (нерегулярные покупатели)</font></li>
</ul>
<p>Покупатели данной группы делают низкие суммарные траты, и их покупки нерегулярны. Их можно привлекать мотивационными скидками и специальными акциями, чтобы увеличить частоту покупок и сделать их более стабильными клиентами.</p>
"""

message_1 = [{'role': 'user',
              'content': 'Write a sales text for buyers who make low total expenses, and their purchases are irregular. They can be attracted with motivational discounts and special promotions to increase the frequency of purchases and make them more stable customers'
            }]

message_2 = [{'role': 'user',
              'content': 'Write a sales text for the product \'Xiaomi Redmi Book 14\' for buyers who make low total expenses, and their purchases are irregular. They can be attracted with motivational discounts and special promotions to increase the frequency of purchases and make them more stable customers'
            }]

output_1 = pipe(message_1, max_new_tokens=300)
output_2 = pipe(message_2, max_new_tokens=300)
response_1 = output_1[0]["generated_text"][-1]["content"].strip()
response_2 = output_2[0]["generated_text"][-1]["content"].strip()
print(response_1, '\n\n')
print(response_2, '\n\n')

"""##  👋  Hey there, value-conscious shopper!  👋

We get it. You're all about **smart spending** and **making every dollar count**.  You're not looking for a big commitment, just **reliable, high-quality products** that fit your unique needs.

**That's why we're here to offer you something special.**

**Introducing our new loyalty program designed for you!**

**Here's how it works:**

* **Earn points on every purchase:**  Every dollar spent earns you points, which can be redeemed for discounts on your next order.
* **Enjoy exclusive promotions:**  Get early access to special deals and discounts just for loyal customers.
* **Free shipping on your next order:**  Reach a certain point threshold and enjoy free shipping on your next purchase.

**Why choose us?**

* **We understand your needs:**  We're all about providing value and flexibility, not overwhelming you with unnecessary commitments.
* **High-quality products:**  We offer a wide range of products that are built to last and meet your specific requirements.
* **Transparent pricing:**  No hidden fees or surprises, just clear and straightforward pricing.

**Ready to unlock the benefits?**

**Sign up for our loyalty program today and start earning points!**

**[Link to sign up]**

**Don't miss out on our special launch offer!**

##  Unlock Productivity Without Breaking the Bank: Redmi Book 14

**Tired of sacrificing quality for affordability?**  The Xiaomi Redmi Book 14 is here to change the game.

**This powerful laptop is packed with features you need, without the price tag that usually comes with them.**  

**Here's why you'll love it:**

* **Unleash your creativity:**  Enjoy a vibrant, high-resolution display perfect for work, school, or creative projects.
* **Effortless performance:**  Get things done quickly with a smooth, responsive processor and ample RAM.
* **Long-lasting battery:**  Work or play for hours on a single charge, without worrying about finding an outlet.
* **Sleek and portable design:**  Take your productivity on the go with a lightweight and stylish laptop.

**And the best part?** We're offering **special promotions** just for you!

**Here's how we can help you make the Redmi Book 14 your new go-to:**

* **Limited-time discounts:**  Get a significant discount on your purchase, making it even more affordable.
* **Flexible payment options:**  Choose a payment plan that fits your budget and make your purchase easier.
* **Loyalty rewards:**  Earn points for every purchase and redeem them for exclusive discounts and deals.

**Don't wait!**  This is your chance to upgrade

##  Unlock Your Savings Potential: Special Offer for Irregular Buyers!

**Offer:**  We know you value your budget, and we want to help you stretch it further.  For a limited time, enjoy **[Discount percentage]% off your next purchase** when you spend [Minimum purchase amount].

**Deadline:** This offer ends on [Date]. Don't miss out on this chance to save!

**Call to Action:**  Shop now and experience the difference!  [Link to your website/shop]

**Why this works:**

* **Motivational Discount:**  The discount is designed to be attractive and encourage immediate action.
* **Targeted Offer:**  The offer is specifically tailored to buyers with low total expenses and irregular purchases, addressing their unique needs.
* **Limited Time:**  The deadline creates a sense of urgency and encourages quick decision-making.
* **Clear Call to Action:**  The call to action is direct and easy to follow, guiding customers to take the next step.


**Additional Tips:**

* **Personalize the message:**  Consider segmenting your audience and tailoring the offer to their specific needs and interests.
* **Highlight the benefits:**  Emphasize how the offer can help them save money and make their purchases more efficient.
* **Use visuals:**  Include images or videos that showcase your products and their benefits.
* **Track your results:**  Monitor the performance of your offer and adjust your

##  Unlock Your Productivity Potential with the Xiaomi Redmi Book 14 -  Limited Time Offer!

**Offer:**  Get **15% off** the Xiaomi Redmi Book 14!

**Deadline:** This offer ends **[Date]**! Don't miss out on this incredible deal.

**Call to Action:**  Upgrade your work experience with the powerful and affordable Xiaomi Redmi Book 14.  Click here to shop now and take advantage of this limited-time offer! [Link to product page]


**Why this works for budget-conscious buyers:**

* **Clear Value Proposition:**  The text emphasizes the product's value proposition - affordability and productivity.
* **Motivational Discount:**  The 15% discount is a significant incentive for buyers who are looking for value.
* **Limited Time Offer:**  The urgency of the deadline creates a sense of scarcity and encourages immediate action.
* **Clear Call to Action:**  The text provides a direct call to action, making it easy for buyers to purchase.


**Additional Tips:**

* **Target your audience:**  Use language that resonates with budget-conscious buyers, emphasizing affordability and practicality.
* **Highlight key features:**  Focus on the Redmi Book 14's key features that are relevant to budget-conscious buyers, such as long battery life, lightweight design, and affordable price.
* **Showcase customer testimonials:**  Include positive customer reviews to build
