import nltk
from nltk.corpus import stopwords
import string
import pandas
import json
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

data_name = []
data_stars = []

data_star_predict = []

with open('business_sample.json', encoding="utf8") as f:
	for line in f:
		json_line = json.loads(line)

		cell_name = json_line.get('name')
		data_name.append(cell_name)

		cell_stars = json_line.get('stars')
		data_stars.append(cell_stars)

dataset = pandas.DataFrame(data={'name': data_name, 'stars': data_stars})

# print(dataset)

# restricting to whole star reviews for increased accuracy:
dataset1 = dataset[(dataset['stars']==1)|(dataset['stars']==2)|(dataset['stars']==3)|(dataset['stars']==4)|(dataset['stars']==5)]
# dataset1 = dataset[(dataset['stars']==1)|(dataset['stars']==3)|(dataset['stars']==5)]
# print(dataset1)
dataset1 = dataset1.dropna()
data = dataset1['name']
target = dataset1['stars'].values
# target = target.astype(int)
# print(target)
# we want to predict stars on the basis of name.

lemmatizer = WordNetLemmatizer()

# need to do pre processing because we have words, not numbers.
def pre_processing(name):
	text_processed = [char for char in name if char not in string.punctuation]
	# so it will process text in names that is not a punctuation mark. (process character for character in text if character is not punctuation mark)
	text_processed =''.join(text_processed)
	# this just puts the processed text together
	return [lemmatizer.lemmatize(word.lower()) for word in text_processed.split() if word.lower() not in stopwords.words('english')]
	# puts word into lowercase if the word is not in the stopwords specified for english.

# now do count vectorizer so you can do text analysis:
print(CountVectorizer(analyzer=pre_processing).fit(data))
count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

with open("count_vectorize_transformer.pickle", "wb") as file:
	pickle.dump(count_vectorize_transformer, file)

data = count_vectorize_transformer.transform(data)
# # this turns it into numbers
# print(data.shape)

# using train test split to split data into training and test groups
data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.25)


machine = MultinomialNB()
machine.fit(data_training, target_training)
predictions = machine.predict(data_test)

print("Text analysis accuracy score:")
print(accuracy_score(target_test, predictions))
print("Text analysis confusion_matrix:")
print(confusion_matrix(target_test, predictions))

with open("serialized_text_analysis.pickle", "wb") as file:
	# some people use .data instead of .pickle
	# must use wb so that you write binary. otherwise it won't work.
	pickle.dump(machine, file)
	# dump() serializes to an open file (file-like object), dumps() serializes to a string.


# # this goes in the programs because it does not require actual stars:

# for row in dataset1['name']:
# 	name_transformed = count_vectorize_transformer.transform([row])
# 	prediction = machine.predict(name_transformed)
# 	# make sure to use the right machine
# 	# print(prediction)
# 	data_star_predict.append(prediction)

# dataset_with_analysis = pandas.DataFrame(data={'name': dataset1['name'],
# 					'predicted_stars': data_star_predict
# 					})

# dataset = pd.concat([dataset1, dataset_with_analysis], axis=1, sort=False)


# for row in dataset1['name']:
# 	name_transformed = count_vectorize_transformer.transform([row])
# 	prediction = machine.predict(name_transformed)
# 	# print(prediction)
# 	data_star_predict.append(prediction)
# dataset4 = pandas.DataFrame(data={'name': dataset3['name'],
# 					'predicted_stars': data_star_predict
# 					})
# print(dataset4)
# dataset = pd.concat([dataset3, dataset4], axis=1, sort=False)

# print(dataset)







