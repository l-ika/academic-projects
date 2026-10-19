import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

data = pd.read_csv('/content/drive/MyDrive/Магистратура/2 курс/СиТАМ/Датасеты/Cuisine_updated.csv')

def indexes(data):
  str = ''
  ind = 1
  for el in data:
    str += '{}{}'.format(el, ', ')
    if ind % 5 == 0:
      str += '\n\t\t\t '
    ind += 1
  return str[:-2]

data.head()

data.info()

row, col = data.shape
print('Количество записей в наборе =', row)
print('Количество признаков в наборе =', col)

print('Список признаков:', indexes(data.columns))

data.isna().sample(50)

msno.matrix(data, color=(0, 0.6, 0.6))

data[(data['cuisine'].isna()) | (data['course'].isna()) | (data['diet'].isna()) | (data['prep_time'].isna())][['name', 'cuisine', 'course', 'diet', 'prep_time', 'ingredients']]

data['name'] = data['name'].str.strip()

data.loc[(data['name'] == 'One Pot Aloo Gobi Sabzi Recipe Using Preethi Electric Pressure Cooker'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Udupi Style Huli Podi Recipe (Sambar Powder)'), ['course', 'diet']] = ['Main Course', 'Vegetarian']
data.loc[(data['name'] == 'Chettinad Rasam Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Homemade Kashmiri Garam Masala Recipe'), 'course'] = 'Spice'
data.loc[(data['name'] == 'Punjabi Garam Masala Powder Recipe'), 'course'] = 'Spice'
data.loc[(data['name'] == 'Brown Rice In Pressure Cooker Method'), 'cuisine'] = 'Indian'
data.loc[(data['name'] == 'Chai Ka Masala Recipe - Homemade Spice Powder for Tea'), ['cuisine', 'course', 'diet', 'prep_time']] = ['Indian', 'Spice', 'Vegetarian', 'Total in 10 M']
data.loc[(data['name'] == 'Ellu Sadam Recipe - Sesame Rice'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Dal Ghiya with Punjabi Wadiyan Recipe - Lentils with Bottle Gourd & Sundried Lentil Dumplings'), ['cuisine', 'course', 'diet', 'prep_time']] = ['Awadhi', 'Lunch', 'Vegetarian', 'Total in 40 M']
data.loc[(data['name'] == 'करी पाउडर रेसिपी - Curry Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'मालवानी मसाला पाउडर रेसिपी - Malvani Masala Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'बिरयानी मसाला पाउडर रेसिपी - Biryani Masala Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Palak Paneer Bhurji Recipe -Spiced Cottage Cheese Scramble With Spinach Recipe'), 'course'] = 'Main Course'
data.loc[(data['name'] == 'चेट्टिनाड मसाला पाउडर रेसिपी - Chettinad Masala Powder Recipe'), 'course'] = 'Spice'
data.loc[(data['name'] == 'Butter Chicken With Tandoori Mayo Recipe'), 'diet'] = 'Non Vegeterian'
data.loc[(data['name'] == 'लहसुनि पुदीना चटनी रेसिपी - Lehsuni Pudina Chutney Recipe'), 'prep_time'] = 'Total in 10 M'
data.loc[(data['name'] == 'Annanas Ka Muzzafar Recipe'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'पालक पनीर भुर्जी रेसिपी - Palak Paneer Bhurji Recipe'), 'course'] = 'Main Course'
data.loc[(data['name'] == 'Malvani Mutton Curry Recipe'), 'prep_time'] = 'Total in 60 M'
data.loc[(data['name'] == 'Chettinad Masala Powder Recipe'), 'course'] = 'Spice'
data.loc[(data['name'] == 'चाट मसाला रेसिपी - Chaat Masala Recipe'), 'course'] = 'Spice'
data.loc[(data['name'] == 'Chaat Masala Recipe'), 'course'] = 'Spice'
data.loc[(data['name'] == 'अचारी मसाला पाउडर रेसिपी - Achari Masala Powder Recipe'), 'course'] = 'Spice'
data.loc[(data['name'] == 'धनसक मसाला पाउडर रेसिपी - Homemade Dhansak Masala Powder Recipe'), 'course'] = 'Spice'
data.loc[(data['name'] == 'Homemade Sambar Powder Recipe-South Indian Curry Powder'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Homemade Dhansak Masala Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Achari Masala Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Mangalorean Style Bafat Pito Recipe-Bafat/Bafad Spice Mix Powder'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Konkani Style Koli Masala Recipe- Spice Mix'), ['course', 'diet', 'prep_time']] = ['Spice', 'Vegetarian', 'Total in 40 M']
data.loc[(data['name'] == 'Maharashtrian Goda Masala Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Panch Phoran Masala Recipe- Indian Five Spice Powder'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Malvani Masala Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Besan Oats Ladoo Recipe'), 'prep_time'] = 'Total in 25 M'
data.loc[(data['name'] == 'Green Tea Coconut Ladoo Recipe - Coconut Energy Balls'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'कढ़ी पकोड़ा रेसिपी - Kadhi Pakoras (Recipe In Hindi)'), 'prep_time'] = 'Total in 55 M'
data.loc[(data['name'] == 'Manga Ogre Recipe (Mango Rice)'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Urulaikizhangu Puli Thokku Recipe (South Indian Style Potatoes with Tamarind)'), ['cuisine', 'course', 'diet', 'prep_time']] = ['South Indian Recipes', 'Main Course', 'Vegetarian', 'Total in 30 M']
data.loc[(data['name'] == 'Classic Pavakkai Stir Fry Recipe (Bitter Gourd Fry)'), ['cuisine', 'course', 'diet', 'prep_time']] = ['South Indian Recipes', 'Lunch', 'Vegetarian', 'Total in 50 M']
data.loc[(data['name'] == 'Menthe Hittu Recipe (Roasted Fenugreek Chutney Powder)'), ['cuisine', 'course', 'diet']] = ['North Karnataka', 'Spice', 'Vegetarian']
data.loc[(data['name'] == 'Tomato Methi Rice Recipe'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Odiya Style Chana Dal With Potato & Pumpkin Curry Recipe'), 'prep_time'] = 'Total in 35 M'
data.loc[(data['name'] == 'Lehsuni Pudina Chutney Recipe'), 'prep_time'] = 'Total in 45 M'
data.loc[(data['name'] == 'Vegetable Uttapam Recipe'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Homemade Methi Ajwain Jeera Powder (Herbal Home Remedy for Digestion)'), 'course'] = 'Spice'
data.loc[(data['name'] == 'वेजिटेबल उत्तपम रेसिपी - Vegetable Uttapam (Recipe In Hindi)'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Peanut Rice With Coconut Recipe'), ['cuisine', 'course', 'diet']] = ['Tamil Nadu', 'Main Course', 'No Onion No Garlic (Sattvic)']
data.loc[(data['name'] == 'टमाटर मेथी पुलाव रेसिपी - Tomato Methi Rice (Recipe In Hindi)'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Khara Obbattu Recipe (Bread Stuffed With Spicy Potato Mixture Recipe)'), ['course', 'diet']] = ['Snack', 'Vegetarian']
data.loc[(data['name'] == 'सब्जी वाली रवा इडली रेसिपी - Veggie Rava Idli (Recipe In Hindi)'), 'prep_time'] = 'Total in 40 M'
data.loc[(data['name'] == 'Kolhapuri Khanda Lasun Masala Spice Mix Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Biryani Masala Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Curry Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Indian Masala Curry Powder With Red Chillies Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'लौकी बड़ी की सब्ज़ी - Lauki Badi Ki Sabzi (Recipe In Hindi)'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Maharashtrian Achar Ka Masala (Pickle Mix Recipe)'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Instant Puliyodharai Powder Recipe (Puliyogare/Pulihora Powder)'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Kolhapuri Masala Recipe (Maharashtrian Masala Chutney Powder)'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Bisi Bele Bath Spice Powder Recipe'), ['course', 'diet']] = ['Spice', 'Vegetarian']
data.loc[(data['name'] == 'Ras El Hanout Recipe (Moroccan Spice Powder)'), 'course'] = 'Spice'
data.loc[(data['name'] == 'Andaman Style Steamed Garlic Prawns Recipe'), 'cuisine'] = 'Indian'
data.loc[(data['name'] == 'Rasgulla Custard Pudding Recipe'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'साउथ इंडियन पापड करी रेसिपी - Appalam Vathal Kuzhambu (Recipe In Hindi)'), 'diet'] = 'Vegetarian'
data.loc[(data['name'] == 'Korean Style Gochujang Recipe (Hot Pepper Paste Recipe)'), 'course'] = 'Spice'
data.loc[(data['name'] == 'Matar Ki Puri (Crispy Puffed Indian Bread With Peas)'), ['course', 'diet']] = ['Snack', 'Vegetarian']

print('Уникальные знечения поля \'cuisine\' = %d наименований' % data['cuisine'].nunique())
print('Уникальные знечения поля \'course\' = %d наименований' % data['course'].nunique())
print('Уникальные знечения поля \'diet\' = %d наименований' % data['diet'].nunique())

print('Значения поля \'cuisine\':', indexes(data['cuisine'].unique()))
print('Значения поля \'course\' :', indexes(data['course'].unique()))
print('Значения поля \'diet\'   :', indexes(data['diet'].unique()))

data.groupby('cuisine')['name'].count().sort_values(ascending=False)
data.groupby('course')['name'].count().sort_values(ascending=False)
data.groupby('diet')['name'].count().sort_values(ascending=False)

data_cuisine = data.groupby('cuisine').size().reset_index(name='count').sort_values(by='count')
data_course = data.groupby('course').size().reset_index(name='count').sort_values(by='count')
data_diet = data.groupby('diet').size().reset_index(name='count').sort_values(by='count')

labels_cuisine = data_cuisine['cuisine']
width_cuisine = data_cuisine['count']

labels_course = data_course['course']
width_course = data_course['count']

labels_diet = data_diet['diet']
width_diet = data_diet['count']

fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(4, 2)

axes_cuisine = fig.add_subplot(gs[0:3, 0:2])
axes_cuisine.barh(labels_cuisine, width_cuisine, color='mediumturquoise')
for index, value in enumerate(data_cuisine['count']):
    axes_cuisine.annotate(value, xy=(value+3, index-0.3), color='dimgray')
axes_cuisine.set_xlabel('Count')
axes_cuisine.set_title('The number of dishes of the cuisine')

axes_course = fig.add_subplot(gs[3, 0])
axes_course.barh(labels_course, width_course, color='lightpink')
for index, value in enumerate(data_course['count']):
    axes_course.annotate(value, xy=(value+7, index-0.2), color='dimgray')
axes_course.set_xlabel('Count')
axes_course.set_title('The number of dishes according to the time of reception')

axes_diet = fig.add_subplot(gs[3, 1])
axes_diet.barh(labels_diet, width_diet, color='lightpink')
for index, value in enumerate(data_diet['count']):
    axes_diet.annotate(value, xy=(value+7, index-0.2), color='dimgray')
axes_diet.set_xlabel('Count')
axes_diet.set_title('The number of dishes depends on the type of diet')

plt.tight_layout()
plt.show()

x_labels = data.groupby(['cuisine', 'course'])['name'].count()
x_labels

data.groupby('prep_time')['name'].count()

data['prep_time'] = data['prep_time'].str.strip()
data['prep_time_num'] = list(map(int, data['prep_time'].str.replace('\t', '').str.replace('Total in ', '').str.partition(' ')[0]))

#Express до 15 мин
#Fast 15 - 30 мин
#Medium 30 - 120 мин
#Long 120 - 240 мин
#Leisurely более 240 мин
data.loc[(data['prep_time_num'] <= 15), 'duration'] = 'Express'
data.loc[((data['prep_time_num'] > 15) & (data['prep_time_num'] <= 30)), 'duration'] = 'Fast'
data.loc[((data['prep_time_num'] > 30) & (data['prep_time_num'] <= 120)), 'duration'] = 'Medium'
data.loc[((data['prep_time_num'] > 120) & (data['prep_time_num'] <= 240)), 'duration'] = 'Long'
data.loc[(data['prep_time_num'] > 240), 'duration'] = 'Leisurely'

data_duration = data.groupby('duration').size().reset_index(name='count').sort_values(by='count')

labels_duration = data_duration['duration']
width_duration = data_duration['count']

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 1)

axes_duration = fig.add_subplot(gs[0:1, 0:1])
axes_duration.barh(labels_duration, width_duration, color='mediumturquoise')
for index, value in enumerate(data_duration['count']):
    axes_duration.annotate(value, xy=(value+15, index-0.07), color='dimgray')
axes_duration.set_xlabel('Count')
axes_duration.set_title('The number of dishes by cooking time')

plt.tight_layout()
plt.show()

ndata = data[['name', 'ingredients']]
ndata['ingredients'] = ndata['ingredients'].str.replace('\t', '').str.replace('\n', '').str.strip()

def clean_ingredients(text):
    text = ''.join([i for i in text if not i.isdigit()])
    text = text.replace('-/', '')
    text = re.sub(r'\s*(cup|cups|teaspoons|teaspoon|tablespoon|tablespoons|gram|grams|inch|bunche|bunches|sprig|pinch|ml|according)', '', text)
    words = word_tokenize(text.lower())
    filter = [word for word in words if word not in stop_words and word not in punctuation]
    return " ".join(filter)

ndata['only ingredients'] = ndata['ingredients'].apply(clean_ingredients)

ndata

def find_dish(name):
  return data[(data['name'] == dish)].index[0]

def recommendation(ind, tfidf_matrix):
  tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

  cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
  cos_sim_data = pd.DataFrame(cos_sim, index=tfidf_data.index, columns=tfidf_data.index)

  similarity_for_find = cos_sim_data[ind].to_frame()
  sorted = similarity_for_find.sort_values(by=ind, ascending=False)
  top = sorted[0:11]
  top_dish = top.index
  return data[['name', 'cuisine', 'course', 'diet', 'duration', 'ingredients']].iloc[top_dish]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(ndata['only ingredients'])

dish = input('Укажите наименование понравившегося блюда - ')
ind = find_dish(dish)
recommendation = recommendation(ind, tfidf_matrix)
recommendation


# С подробным выводом

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(ndata['only ingredients'])

tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_data

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cos_sim_data = pd.DataFrame(cos_sim, index=tfidf_data.index, columns=tfidf_data.index)
cos_sim_data

similarity_for_find = cos_sim_data[ind].to_frame()
sorted = similarity_for_find.sort_values(by=ind, ascending=False)
sorted

top = sorted[1:11]
 top

top_dish = top.index
data[['name', 'cuisine', 'course', 'diet', 'duration', 'ingredients']].iloc[top_dish]
