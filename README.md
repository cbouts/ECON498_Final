# ECON498 Final Project

## Contents
- [Introduction](#introduction)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
  * [PART A](#part-a)
  * [OPTIONAL: Train text analysis tools](#optional-train-text-analysis-tools)
  * [Step 1: Import packages](#step-1-import-packages)
  * [Step 2: Prepare structures](#step-2-prepare-structures)
    + [2- Modifications](#2-modifications)
  * [Step 3: Construct for loop](#step-3-construct-for-loop)
    + [3- Modifications](#3-modifications)
  * [Step 4: Collect business info and run text analysis for relevant businesses](#step-4-collect-business-info-and-run-text-analysis-for-relevant-businesses)
  * [Step 5: Location](#step-5-location)
    + [5- Modifications](#5-modifications)
  * [Step 6: Hours](#step-6-hours)
    + [6- Modifications](#6-modifications)
  * [Step 7: Attributes](#step-7-attributes)
    + [7- Modifications](#7-modifications)
  * [Step 8: Categories](#step-8-categories)
    + [8- Modifications](#8-modifications)
  * [Step 9: Make the DataFrame](#step-9-make-the-dataframe)
    + [9- Modifications](#9-modifications)
  * [PARTS B AND C](#parts-b-and-c)
  * [Step 10: Train models](#step-10-train-models)
  * [Step 11: Make dataframe from the no-stars file](#step-11-make-dataframe-from-the-no-stars-file)
  * [Step 12: Make predictions](#step-12-make-predictions)
  * [Step 13: Run the programs](#step-13-run-the-programs)
- [See also: report.pdf](#see-also-reportpdf)
  
    
## Introduction
This project trains two random forest machines to predict review scores for restaurant and non-restaurant businesses on the basis of information found on these businesses' [Yelp](yelp.com) pages. It then uses this machine to predict review scores for businesses not found on the training .json file.

## Technologies
This project was made on Python 3.7, so that or a newer version may be required to run it. 

The programs also use the modules numpy, pandas, sklearn, time, pickle, and nltk. If you don't have these already, you may need to install them by writing:
`pip -m install numpy, pandas, sklearn, time, pickle, and nltk`
in the terminal.

## Installation
To run this project, you will need to install it on your computer. To do this, write
`pip install git+git://github.com/cbouts/ECON498_Final.git`
in the terminal.

## Usage
The project contains one file which is optional to use ([try_text_analysis.py](https://github.com/cbouts/ECON498_Final/blob/main/try_text_analysis.py)) and two main python files: [runme_restaurants.py](https://github.com/cbouts/ECON498_Final/blob/master/runme_restaurants.py) and [runme_nonrestaurants.py](https://github.com/cbouts/ECON498_Final/blob/main/runme_nonrestaurants.py).
The runme files predict review scores for restaurants and non-restaurant businesses, respectively, while the text analysis file trains a count vectorizer and a Naive Bayes text analysis machine for later use in the runmes.

The steps listed and elaborated below explain the processes in the programs and advise users on modifying the programs to fit their needs. 

Because the programs do the same things with different information reflecting differences in information relevant to restaurants and non-restaurant businesses, we use only one of the files (runme_nonrestaurants.py) to explain the relevant processes. 

The two runme files are not run until the final step.

### PART A

### OPTIONAL: Train text analysis tools
I use text analysis in my programs to strengthen my predictions. This is incorporated in both runme files. Both runme files make use of two serialized files [count_vectorize_transformer.pickle](https://github.com/cbouts/ECON498_Final/blob/master/count_vectorize_transformer.pickle) and [serialized_text_analysis.pickle](https://github.com/cbouts/ECON498_Final/blob/master/serialized_text_analysis.pickle). These have been already been constructed by running [try_text_analysis.py](https://github.com/cbouts/ECON498_Final/blob/master/try_text_analysis.py). 

Since both serialized files are included in this project, there is no real need to re-run  [try_text_analysis.py](https://github.com/cbouts/ECON498_Final/blob/master/try_text_analysis.py). However, if you really want to, you can do it.

Here, I explain the contents of [try_text_analysis.py](https://github.com/cbouts/ECON498_Final/blob/master/try_text_analysis.py).

After importing the needed packages...
```
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
```
... the program creates empty containers to hold the information we get from the training dataset, business_sample.json:
```
data_name = []
data_stars = []
data_star_predict = []
```
Next, the program opens the training dataset business_sample.json and accounts for the fact that this is a unicode(UTF-8) file:
```
with open('business_sample.json', encoding="utf8") as f:
```
... and creates the for loop that will read and collect the business name and star count in each line of business_sample.json:
```
with open('business_sample.json', encoding="utf8") as f:
	for line in f:
		json_line = json.loads(line)

		cell_name = json_line.get('name')
		data_name.append(cell_name)

		cell_stars = json_line.get('stars')
		data_stars.append(cell_stars)
```
Having done this, the program appends the information it has gathered to a dataframe:
```
dataset = pandas.DataFrame(data={'name': data_name, 'stars': data_stars})
```
The program is almost ready to begin text analysis machine training. Before it can do that, though, it must make some changes to the dataframe. 
```
dataset1 = dataset[(dataset['stars']==1)|(dataset['stars']==2)|(dataset['stars']==3)|(dataset['stars']==4)|(dataset['stars']==5)]

dataset1 = dataset1.dropna()
```
The first line in the above code restricts the training data to include only those businesses with whole-star ratings. We include this because the text analysis program has a hard time distinguishing between, for example, 1.5 and 1 or 2, but a much easier time distinguishing between 2 and 1 or 3. The second line drops rows in the dataset containing empty values.

Now, we get to the crux of the program. We first communicate to the computer that we would like it to predict review star count on the basis of the name of the business:
```
data = dataset1['name']
target = dataset1['stars'].values
```
We include `.values` in the target because this is necessary for the machine we construct later in the program.

Next, the program defines a pre-processing method for the text in each business name. The comments in this code explain what each line does.
```
lemmatizer = WordNetLemmatizer()

# need to do pre processing because we have words, not numbers.
def pre_processing(name):
	text_processed = [char for char in name if char not in string.punctuation]
	# so it will process text in names that is not a punctuation mark. (process character for character in text if character is not punctuation mark)
	text_processed =''.join(text_processed)
	# this just puts the processed text together
	return [lemmatizer.lemmatize(word.lower()) for word in text_processed.split() if word.lower() not in stopwords.words('english')]
	# puts word into lowercase if the word is not in the stopwords specified for english.
```
We next use this pre processor to construct a count vectorize transformer.
```
count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)
```
We serialize the count vectorize transformer for later use in the runme files. The count vecotrizer becomes [count_vectorize_transformer.pickle](https://github.com/cbouts/ECON498_Final/blob/master/count_vectorize_transformer.pickle).
```
with open("count_vectorize_transformer.pickle", "wb") as file:
	pickle.dump(count_vectorize_transformer, file)
```
It's now time to train the text analysis machine that becomes [serialized_text_analysis.pickle](https://github.com/cbouts/ECON498_Final/blob/master/serialized_text_analysis.pickle). We use the count vectorize transformer to prepare the data for use in the training (remember, we have already defined data: `data = dataset1['name']`):
```
data = count_vectorize_transformer.transform(data)
```
We use train test split to split the training dataset into four categories: data_training, target_training, data_test, target_test. This allows us to fit a Multinomial Naive Bayes machine with data_training and target_training, and then predict target_test on the basis of data_test. 
```
data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.25)
machine = MultinomialNB()
machine.fit(data_training, target_training)
predictions = machine.predict(data_test)
```
Since it's a supervised learning model and we actually have target_test, we can use accuracy score and confusion matrix to show how well our predictions get at target_text. We print out accuracy score and confusion matrices.
```
print("Text analysis accuracy score:")
print(accuracy_score(target_test, predictions))
print("Text analysis confusion_matrix:")
print(confusion_matrix(target_test, predictions))
```
Finally, we serialize the machine we've trained into [serialized_text_analysis.pickle](https://github.com/cbouts/ECON498_Final/blob/master/serialized_text_analysis.pickle) for use in the runme files. 
```
with open("serialized_text_analysis.pickle", "wb") as file:
	pickle.dump(machine, file)
```


### Step 1: Import packages
We do this with the code:
```
import pandas
import json
import ast
from sklearn.impute import SimpleImputer
import numpy as np
from statistics import mode
from statistics import mean
from datetime import datetime
import time
import kfold_template
import pickle
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import string
```

### Step 2: Prepare structures
Create empty structures to contain the information we will take from the .json file.
This is done with:
```
data_name = []
data_prediction = []
data_address = []
data_city = []
...
data_events = []
data_nightlife = []
```
In this step, we prepare to collect information about businesses' names, locations, hours, attributes (such as whether or not the business accepts insurance), parking availabilities, and sectors (categories). 

[runme_restaurants.py](https://github.com/cbouts/ECON498_Final/blob/master/runme_restaurants.py) includes more attributes, but the concept is the same. Also, [runme_restaurants.py](https://github.com/cbouts/ECON498_Final/blob/master/runme_restaurants.py)'s categories reflect different types of restaurants, not different business sectors. 

#### 2- Modifications:
If you find additional information on the .json file that you would like to include in your analysis, you can create additional structures to hold this information. For example, you can add `data_new_information = []`. Alternatively, you can remove structures whose information you are not interested in. However, if you do this, you will need to remove all occurences of `data_removed_in_modification` in the rest of the code. It's really not necessary to remove these structures here because you can just delete or comment out the code associated with them later in the file.

### Step 3: Construct for loop
Open and read the .json file line by line, accounting for the fact that business_sample.json (the training dataset) uses utf8 encoding. 
```
with open('business_sample.json', encoding="utf8") as f:
	for line in f:
		json_line = json.loads(line)
```
This creates the for loop that will be used to gather the rest of the information for training our machines from the .json training file. It reads line by line so that the program will gather all the information about one business before moving on to the next business.

#### 3- Modifications:
If you would like to run the program with a different .json training file, just replace the file name in the original line of code and ensure that the new file matches the structure of business_sample.json.

#### Step 4: Collect business info and run text analysis for relevant businesses

Read from the json line to collect business name and business categories. Append the categories to "list_categories." If categories are listed for the business, append "1" (meaning "true") to "data_restaurant" if "restaurant" is a category listed for the business and "0" (meaning "false") if the categories do not include "restaurant." Otherwise, append an empty value (None) to data_restaurant. 

```
		cell_name = json_line.get('name')
		data_name.append(str(cell_name))

		# # To get the stars score
		cell_stars = json_line.get('stars')
		data_stars.append(cell_stars)

		cell_categories = json_line.get('categories')
		list_categories.append(cell_categories)

		if cell_categories is not None:
			cell_restaurant = (1 * ('Restaurants' in cell_categories))
			data_restaurant.append(cell_restaurant)
		else:
			data_restaurant.append(None)
```

Now that we know whether or not the business is a restaurant, we can do text analysis for only restaurants in [runme_restaurants.py](https://github.com/cbouts/ECON498_Final/blob/master/runme_restaurants.py) and only non-restaurant businesses in [runme_nonrestaurants.py](https://github.com/cbouts/ECON498_Final/blob/master/runme_nonrestaurants.py). 

To do this, we make an "if-else" structure beginning with `if cell_restaurant != 1:` in [runme_nonrestaurants.py](https://github.com/cbouts/ECON498_Final/blob/master/runme_nonrestaurants.py) and `if cell_restaurant == 1:` in [runme_restaurants.py](https://github.com/cbouts/ECON498_Final/blob/master/runme_restaurants.py).

This cuts down the time that each program spends on unnecessary text analysis. Since the code we use for this example comes from the non-restaurant runme, we perform text analysis only those businesses which do not include "restaurant" as a category. The comments in the code below explain the text analysis process. For an alternate explanation of the process, read through [OPTIONAL: Train text analysis tools](#optional-train-text-analysis-tools). 

```
		if cell_restaurant != 1:

			lemmatizer = WordNetLemmatizer()
			# lemmatizes words for text analysis

			# We need to do pre processing because the names of the businesses consist of text, not numbers.
			def pre_processing(text):
				text_processed = [char for char in text if char not in string.punctuation]
				# this sets it up to process every character that is not a punctuation mark. 
				text_processed =''.join(text_processed)
				# this puts the processed text together
				return [lemmatizer.lemmatize(word.lower()) for word in text_processed.split() if word.lower() not in stopwords.words('english')]
				# puts each word into lowercase if the word is not in the stopwords specified for english.

			# after this, the dataset becomes a bunch of words waiting for you to do the analysis, which you still can't really do because you need the count vectorizer.
			transformer = open("count_vectorize_transformer.pickle", "rb")
			# this opens the serialized count vectorizer that I trained in another program.

			count_vectorize_transformer = pickle.load(transformer)
			# this accesses the count vectorizer

			cell_name = json_line.get('name')
			text = [cell_name]

			text_transformed = count_vectorize_transformer.transform(text)
			# this transforms the text in the business name into something that can be analyzed with text analysis.

			file = open("serialized_text_analysis.pickle", "rb")
			# this opens the serialized Naive Bayes text analysis machine that I trained in another program.

			machine = pickle.load(file)
			# this accesses the Naive Bayes text analysis machine I trained earlier.

			cell_prediction = machine.predict(text_transformed)
			# this predicts the star count for the current business by analysing the text in the business's name.

			cell_prediction = np.mean(cell_prediction)
			data_prediction.append(cell_prediction)
			# this appends the predicted star count to data_prediction.

			print(text)
			print(" ------------------------------------ okay")

		else:
			data_prediction.append(None)
			# this appends an empty value for data_prediction for restaurants in the case of runme_nonrestaurants.py and an empty value for data_prediction for non-restaurant businesses in the case of runme_restaurants.py. 
```

### Step 5: Location
Get location data for the given business. Append city and address...
```
		# # To get the address of the business
		cell_address = json_line.get('address')
		data_address.append(cell_address)

		cell_city = json_line.get('city')
		data_city.append(cell_city)
```

... then construct dummy variables for 15 most common cities in the data, equal to 1 if the `cell_city == x` statement is true and 0 if it's false...
```
		data_vegas.append(1 * (cell_city == "Las Vegas"))
		data_toronto.append(1 * (cell_city == "Toronto"))
		data_phoenix.append(1 * (cell_city == "Phoenix"))
		data_clt.append(1 * (cell_city == "Charlotte"))
		data_scottsdale.append(1 * (cell_city == "Scottsdale"))
		data_calgary.append(1 * (cell_city == "Calgary"))
		data_pitt.append(1 * (cell_city == "Pittsburgh"))
		data_montreal.append(1 * (cell_city == "Montréal"))
		data_mesa.append(1 * (cell_city == "Mesa"))
		data_henderson.append(1 * (cell_city == "Henderson"))
		data_tempe.append(1 * (cell_city == "Tempe"))
		data_chandler.append(1 * (cell_city == "Chandler"))
		data_cleveland.append(1 * (cell_city == "Cleveland"))
		data_glendale.append(1 * (cell_city == "Glendale"))
		data_madison.append(1 * (cell_city == "Madison"))
```
... then append the business's state...
```
		cell_state = json_line.get('state')
		data_state.append(cell_state)
```
... and create dummy variables for all states which are represented by over 900 businesses in the training data. (We don't include the other states that are represented in the training data because each of them is represented by under 20 businesses, so we would have strong reason to doubt the validity of predictions on the basis of belonging to these states.) For example: 
```
		if cell_state == "AB":
			data_AB.append(1)
		else:
			data_AB.append(0)
```

We use state data to create dummies for the three countries represented in the training data: the US, the UK, and Canada. Very few UK businesses are represented in the training data, so we do append the UK dummy to the dataframe used for predicting stars later in the program (we don't make predictions on the basis of belonging or not belonging to the UK). However, we include the UK here because it makes the process of forming data_US much simpler:
```
		if cell_state == "AB" or cell_state == "BC" or cell_state == "ON" or cell_state == "QC":
			cell_CAN = 1
			data_CAN.append(1)
		else:
			cell_CAN = 0
			data_CAN.append(0)

		if cell_state == "CON" or cell_state == "BAS" or cell_state == "DOW" or cell_state == "DUR" or cell_state == "XGL" or cell_state == "XGM" or cell_state == "XWY":
			cell_UK = 1
			data_UK.append(1)
		else:
			cell_UK = 0
			data_UK.append(0)

		if cell_CAN == 0 and cell_UK == 0:
			data_US.append(1)
		else:
			data_US.append(0)
```

Finally, append latitude and longitude of the business.
```
		cell_latitude = json_line.get('latitude')
		data_latitude.append(cell_latitude)

		cell_longitude = json_line.get('longitude')
		data_longitude.append(cell_longitude)
```

#### 5- Modifications:
If you are interested in creating a dummy variable for business locations in other cities, you can add more cities in the same way that I did. Make sure to also create structures for these (like in step 2) and include them in the dataframe that you will use for analysis later in the program.

### Step 6: Hours
Get information about business hours.

Append hours from the .json file. Then use the cell_hours (the list of hours for the entire week) to find hours for each day of the week:
```
		# # To get hours
		cell_hours = json_line.get('hours')
		data_hours.append(cell_hours)

		# getting hours for each day
		if cell_hours is not None:
			cell_monday = cell_hours.get('Monday') 
			cell_tuesday = cell_hours.get('Tuesday')
			cell_wednesday = cell_hours.get('Wednesday')
			cell_thursday = cell_hours.get('Thursday')
			cell_friday = cell_hours.get('Friday')
			cell_saturday = cell_hours.get('Saturday')
			cell_sunday = cell_hours.get('Sunday')
		else:
			cell_monday = None
			cell_tuesday = None
			cell_wednesday = None
			cell_thursday = None
			cell_friday = None
			cell_saturday = None
			cell_sunday = None
		data_monday.append(cell_monday)
		data_tuesday.append(cell_tuesday)
		data_wednesday.append(cell_wednesday)
		data_thursday.append(cell_thursday)
		data_friday.append(cell_friday)
		data_saturday.append(cell_saturday)
		data_sunday.append(cell_sunday)
```

We also get opening and closing times for each day of the week. This is fairly complicated, but the comments in the code explain how it works.

We use Monday as an example:
```
		if cell_monday is not None:
			cell_monday_close = pandas.to_timedelta((cell_monday.split('-')[1]) + ":00").total_seconds()
			# cell_monday.split('-')[1] accesses the raw data for closing time on monday.
			# + ":00" adds seconds to the closing time
			# pandas.to_timedelta() converts Monday's closiing time to a timedelta format which we can use for addition and subtraction later in the program.
			# total_seconds() converts the closing time to seconds, which makes it easier to do addition and subtraction later in the program.
			if cell_monday_close == 0.0:
				# this is what happens if the closing time is recorded as 0:00 in the .json file. 
				# This means that the closing time is midnight, so we need to make sure that the data clearly reflects the fact that the business is open until midnight.
				# leaving the closing time as 0 would cause problems for addition and subtraction later (being open from 0 to 0 means that the business is open 24 hours), so we change the closing time to 1 second less than midnight:
				cell_monday_close = 86399.0
			else:
				cell_monday_close = cell_monday_close
			cell_monday_open = pandas.to_timedelta((cell_monday.split('-')[0]) + ":00").total_seconds()
			if cell_monday_open == 0.0:
				# this is what happens if the opening time is recorded as 0:00 in the .json file. 
				# This means that the opening time is midnight, so we need to make sure that the data clearly reflects the fact that the business opens at midnight.
				# leaving the opening time as 0 would cause problems for addition and subtraction later, so we change the closing time to 1 second after midnight:
				cell_monday_open = 1.0
			else:
				cell_monday_open = cell_monday_open
		else:
			# we record both open and close times as 0 for businesses that are closed on the given day. 0 - 0 is 0 total seconds open in the day.
			cell_monday_close = 0
			cell_monday_open = 0
		data_monday_close.append(cell_monday_close)
		data_monday_open.append(cell_monday_open)
```
We use opening and closing times for each day to calculate total hours per day of the week for each business (again using Monday as an example):
```
		if cell_monday is not None:
			if cell_monday_open < cell_monday_close:
				cell_monday_total_hours = (cell_monday_open - cell_monday_close)
			elif cell_monday_open > cell_monday_close:
				cell_monday_total_hours = (cell_monday_close - cell_monday_open)
			else:
				cell_monday_total_hours = 0
		else:
			cell_monday_total_hours = 0
		data_monday_total_hours.append(cell_monday_total_hours)
```
We determine how many days per week the business is open using the knowledge that `cell_hours == None` means the business is closed on a given day:
```
		if cell_hours is not None:
			if cell_monday is None:
				cell_days_open_after_mon = 6
			else:
				cell_days_open_after_mon = 7

			if cell_tuesday is None:
				cell_days_open_after_tues = (cell_days_open_after_mon - 1)
			else:
				cell_days_open_after_tues = cell_days_open_after_mon

			if cell_wednesday is None:
				cell_days_open_after_weds = (cell_days_open_after_tues - 1)
			else:
				cell_days_open_after_weds = cell_days_open_after_tues

			if cell_thursday is None:
				cell_days_open_after_thurs = (cell_days_open_after_weds - 1)
			else:
				cell_days_open_after_thurs = cell_days_open_after_weds

			if cell_friday is None:
				cell_days_open_after_fri = (cell_days_open_after_thurs - 1)
			else:
				cell_days_open_after_fri = cell_days_open_after_thurs

			if cell_saturday is None:
				cell_days_open_after_sat = (cell_days_open_after_fri - 1)
			else:
				cell_days_open_after_sat = cell_days_open_after_fri

			if cell_sunday is None:
				cell_days_open = (cell_days_open_after_sat - 1)
			else:
				cell_days_open = cell_days_open_after_sat
		else:
			cell_days_open = 0
		data_days_open.append(cell_days_open)
```
We also use the information we've gathered about hours to construct some new information. 

We are interested in finding out whether or not the business is open at all on weekends:
```
		# to see if the business is open at all on weekends:
		if cell_sunday is None and cell_saturday is None:
			data_weekends.append(0)
		else: 
			data_weekends.append(1)
```
... determining the the business's most common opening and closing times using the `mode` function:
```
		if cell_hours is not None:
			cell_open_mode = mode([cell_monday_open, cell_tuesday_open, cell_wednesday_open, cell_thursday_open, cell_friday_open, cell_saturday_open, cell_sunday_open])
			data_open_mode.append(cell_open_mode)
		else:
			data_open_mode.append(0)

		if cell_hours is not None:
			cell_close_mode = mode([cell_monday_close, cell_tuesday_close, cell_wednesday_close, cell_thursday_close, cell_friday_close, cell_saturday_close, cell_sunday_close])
			data_close_mode.append(cell_close_mode)
		else:
			data_close_mode.append(0)
```
... calculating the total hours open per day (in seconds). (This yields negative values because we use timedeltas, but because this preserves the relationship between hours per workday and ratings, it is okay.) We continue to use Monday as an example:
```
		# getting total hours per workday in seconds. it gives us negative values because it is a timedelta, but this is fine as it preserves a relationship between rating stars and total hours open.
		if cell_monday is not None:
			if cell_monday_open < cell_monday_close:
				cell_monday_total_hours = (cell_monday_open - cell_monday_close)
			elif cell_monday_open > cell_monday_close:
				cell_monday_total_hours = (cell_monday_close - cell_monday_open)
			else:
				cell_monday_total_hours = 0
				# total hours for a business that has identical opening and closing times in seconds is closed on the given day.
		else:
			cell_monday_total_hours = 0
			# total hours for a business that has no open time on monday is 0
		data_monday_total_hours.append(cell_monday_total_hours)
```
... and getting mean hours per day and total hours per week:
```
# to get the mean hours per day:
		if cell_hours is not None:
			cell_mean_hours = mean([cell_monday_total_hours, cell_tuesday_total_hours, cell_wednesday_total_hours, cell_thursday_total_hours, cell_friday_total_hours, cell_saturday_total_hours, cell_sunday_total_hours])
			data_mean_hours.append(cell_mean_hours)
		else:
			data_mean_hours.append(0)

# to get the total hours per week
		if cell_hours is not None:
			cell_total_hours = (cell_monday_total_hours + cell_tuesday_total_hours + cell_wednesday_total_hours + cell_thursday_total_hours + cell_friday_total_hours + cell_saturday_total_hours + cell_sunday_total_hours)
			data_total_hours.append(cell_total_hours)
		else:
			data_total_hours.append(0)
```

#### 6- Modifications:
You can use the .json data about hours to create additional indicators about business hours in the same way that I did if you are interested in them. If you do this, make sure to also create structures for these (like in step 2) and include them in the dataframe that you will use for analysis later in the program.

### Step 7: Attributes
Next, we collect information about business attributes from the .json file.

First, get the list of attributes for the given business:
```
		cell_attribute = json_line.get('attributes')
		list_attributes.append(cell_attribute)
```

Then use `cell_attribute` to gather information about the business. There are many attributes, but we use "Accepts credit cards" as an example:
```
		if cell_attribute is not None:
			cell_credit_cards = cell_attribute.get('BusinessAcceptsCreditCards')
			# this makes cell_credit_cards a boolean variable that takes values "true" or "false"
			if cell_credit_cards is not None:
				if cell_credit_cards == True:
					data_credit_cards.append(1)
				else:
					data_credit_cards.append(0)
			else:
				data_credit_cards.append(0)
		else:
			data_credit_cards.append(None)
```

Information about parking is listed in attributes. We do the same process for parking information, but because it is nested inside the "parking" attribute, the process is slightly different:

```
		if cell_attribute is not None:
			cell_parking = cell_attribute.get('BusinessParking')
			if cell_parking is not None:
				x = ast.literal_eval(cell_parking)
				# we construct x to allow us to read through the information in the cell_parking attribute because the cell_parking attribute is not itself a true/false variable, but a container of other true/false variables.
				if x is not None:
					try:
						cell_garage = x['garage']
						# locating 'garage' within the cell_parking attribute
						if cell_garage is not None:
							data_garage.append(1 * cell_garage)
						else:
							data_garage.append(0)
					except KeyError:
						# handling any errors that arose during the process
						data_garage.append(None)

					try:
						cell_streetparking = x['street']
						if cell_streetparking is not None:
							data_streetparking.append(1 * cell_streetparking)
						else:
							data_streetparking.append(0)
					except KeyError:
						data_streetparking.append(None)

					try:
						cell_validated = x['validated']
						if cell_validated is not None:
							data_validated.append(1 * cell_validated)
						else:
							data_validated.append(0)
					except KeyError:
						data_validated.append(None)

					try:
						cell_lot = x['lot']
						if cell_lot is not None:
							data_lot.append(1 * cell_lot)
						else:
							data_lot.append(0)
					except KeyError:
						data_lot.append(None)

					try:
						cell_valet = x['valet']
						if cell_valet is not None:
							data_valet.append(1 * cell_valet)
						else:
							data_valet.append(0)
					except KeyError:
						data_valet.append(None)

					if data_garage == 1 or data_streetparking == 1 or data_validated == 1 or data_lot == 1:
						data_regular_parking.append(1)
					elif data_garage == None and data_streetparking == None and data_validated == None and data_lot == None:
						data_regular_parking.append(None)
					else:
						data_regular_parking.append(0)
				else:
					#  if there is no parking attribute, we append empty values for each of the indicators.
					data_garage.append(None)
					data_streetparking.append(None)
					data_validated.append(None)
					data_lot.append(None)
					data_valet.append(None)
					data_regular_parking.append(None)
			else:
				#  if there is no parking attribute, we append empty values for each of the indicators.
				data_garage.append(None)
				data_streetparking.append(None)
				data_validated.append(None)
				data_lot.append(None)
				data_valet.append(None)
				data_regular_parking.append(None)
		else:
			# if there are no attributes listed for the business, we append empty values for each indicator.
			data_garage.append(None)
			data_streetparking.append(None)
			data_validated.append(None)
			data_lot.append(None)
			data_valet.append(None)
			data_regular_parking.append(None)
```

runme_restaurants.py contains many different attributes, but the processes for regular attributes (like "Accepts credit cards") and nested attributes (like "Parking) are the same as those shown here. 

#### 7- Modifications:

You can use the .json data about attributes to include other attributes in the same way that I did if you are interested in them. If you do this, make sure to also create empty structures for these (like in step 2) and include them in the dataframe that you will use for analysis later in the program.


### Step 8: Categories

We have already found a list of categories in step 1. We now call `cell_categories` to learn more about the business's categories and create dummy variables for each possible category. We use Health & Medical as an example:
```
		if cell_categories is not None:
			cell_medical = (1 * ('Health & Medical' in cell_categories))
			data_medical.append(cell_medical)
		else:
			data_medical.append(None)
```
While categories indicate business sectors in non-restaurant data, restaurant data (handled by runme_restaurants.py) has categories to indicate cuisine. The processes of finding and appending category information are the same in each program. 


#### 8- Modifications:

Again, you can use the .json data about categories to include other attributes in the same way that I did if you are interested in them. If you do this, make sure to also create empty structures for these (like in step 2) and include them in the dataframe that you will use for analysis later in the program.


### Step 9: Make the DataFrame

We use the variables constructed in step 2 to append the information we are interested in using for predictions to a pandas DataFrame called "dataset". Later, machines can use information from "dataset" to make predictions about review scores in the same way that they would read and use information from a CSV. 

It is very important in this step that we comment out the variables (with the exception of 'name') which contain non-float/integer objects because the models we train later don't accept them.

```
dataset = pandas.DataFrame(data={'name': data_name,
					'text_analysis_prediction': data_prediction,
					
					...
					
					
					# 'categories_list': list_categories,
					'is_restaurant': data_restaurant,
					'medical': data_medical,
					'beauty': data_beauty,
					'shopping': data_shopping,
					'local_services': data_local_services,
					'travel': data_travel,
					'arts_entertainment': data_arts_entertainment,
					'home_services': data_home_services,
					'professional_services': data_professional_services,
					'automotive': data_automotive,
					'education': data_education,
					'fitness_and_instruction': data_fitness_and_instruction,
					'events': data_events,
					'nightlife': data_nightlife
						})
```

Having constructed "dataset", we now restrict the dataset to businesses which do not list "restaurant" as a category (note that for runme_nonrestaurants.py we use `!=` while in runme_restaurants.py we use `==`):

```
dataset = dataset[(dataset['is_restaurant']!=1)]
```
... and drop rows containing empty values and "None" strings:
```
dataset = dataset.dropna()
dataset = dataset.replace(to_replace='None',value=np.nan).dropna()
```

The dataset is now ready to be used for training our models!! 

#### 9- Modifications:
If you have made modifications in steps 2-8, make sure that these are reflected in your DataFrame. Add any new variables that you have added to the program, or comment out any that you are not interested in. 

### Step 10: Train models

Now it's time to train our models. We are mostly interested in training and using a random forest classifier, but for comparison, we also train a linear model in this step. 

The program is almost ready to begin model training. Before it can do that, though, it must make some changes to the dataframe. First, it restricts the dataset to nonrestaurant businesses in runme_nonrestaurants.py with `dataset = dataset[(dataset['is_restaurant']!=1)]` and restaurantsin runme_restaurants.py with `dataset = dataset[(dataset['is_restaurant']==1)]`.

It then drops missing values, including strings that say "None":
```
dataset = dataset.dropna()
dataset = dataset.replace(to_replace='None',value=np.nan).dropna()
```

To improve the accuracy of our machine, we next restrict the training data to include only those businesses with whole-star ratings. This is because our machines have a hard time distinguishing between, for example, 1.5 and 1 or 2, but a much easier time distinguishing between 2 and 1 or 3.
```
dataset1 = dataset[(dataset['stars']==1)|(dataset['stars']==2)|(dataset['stars']==3)|(dataset['stars']==4)|(dataset['stars']==5)]
```
Now, we begin training our models. We first communicate to the computer that we would like it to predict review star count (the information in the column indexed "1" in the dataframe) on the basis the information in the other colunms of the dataset. We exclude "name" from the data because it is a string, not a float or integer value that can be handled by our models. 
```
target = dataset.iloc[:,1].values
data = dataset.iloc[:,2:].values
```
We include `.values` in the data and target because this is necessary for the machine we construct later in the program.

It's now time to train the random forest machine. We use train test split to split the training dataset into four categories: data_training, target_training, data_test, target_test. This allows us to fit a Random Forest Classifier machine with data_training and target_training, and then predict target_test on the basis of data_test. For the moment, ignore the numbers in `n_estimators=` and `max_depth`. Just know that `n_estimators=` is the number of trees that the Random Forest constructs, and `max_depth` is the maximum depth of any given tree in the Random Forest.
```
data_training, data_test, target_training, target_test = train_test_split(data, target, test_size=0.25)
machine = RandomForestClassifier(n_estimators=800, criterion="gini", max_depth=17)
machine.fit(data_training, target_training)
prediction = machine.predict(data_test)
```
Since it's a supervised learning model and we actually have target_test, we can use accuracy score and confusion matrix to show how well our predictions get at target_text. We print out accuracy score and confusion matrices.
```
print("Confusion matrix from nonrestaurant random forest:")
print(confusion_matrix(target_test, prediction))

print("Accuracy score from nonrestaurant random forest:")
print(accuracy_score(target_test, prediction))
```
Now, remember `n_estimators` and `max_depth`. These parameters obviously influence the accuracy scores and confusion matrices we reach with our classifier, but it's impossible to know the best values for these parameters unless we use trial and error. That means running the program repeatedly with different specifications of these parameters until we have optimized the accuracy score our machine gives us. I track my trial and error here:
```
# 200 and 25 gave .51
# 50 and 10 gave .46
# 200 and 30 gave .525
# 400 and 30 gave .48
# 200 and 33 gave .5
# 200 and 29 gave .5
# 1000 and 30 gave .51
# 200 and 15 gave .509
# 250 and 30 gave .514
# 150 and 30 gave .507
# 225 and 30 gave .50
# 300 and 15 gave .527
# 250 and 35 gave .489
# 500 and 15 gave .530
# 500 and 16 gave .538
# 800 ad 16 gave .55
# 800 and 17 gave .556
# 800 and 18 gave .51
# 1000 and 17 gave .517

# so I use 800 and 17!
```
Having found the best accuracy score with `n_estimators=800` and `max_depth=17`, I set these parameters in my machine for the next step of the program- serializing the machine for future use in prediction of unknown star counts.
```
with open("nonrestaurants_serialized_random_forest_text.pickle", "wb") as file:
	pickle.dump(machine, file)
```
We now train the Linear Regression model for comparison. We construct the machine and use data_training, target_training, data_test, and target_test from the train test split we used to train the random forest.
```
machine = linear_model.LinearRegression()
machine.fit(data_training, target_training)
```
Next, because we don't manipulate parameters for linear regression, we immediately serialize this fitted machine for potential future use:
```
with open("nonrestaurants_serialized_linear_text.pickle", "wb") as file:
	pickle.dump(machine, file)
```
We use the linear regression to predict the data_test with `prediction = machine.predict(data_test)`. Since this is a supervised learning model and we actually have target_test, we can use r2 score to show how predictions by the machine get at target_text. We print out r2 score for use in later discussion/comparison.
```
print("r2 score from nonrestaurant linear model:")  
print(r2_score(target_test, prediction))
```
### PARTS B AND C

### Step 11: Make dataframe from the no-stars file

We have trained our models and we are now ready to prepare to make predictions on the basis of no-star business data.

Luckily, the preparations for this are nearly identical to those listed in steps 2-9. However, if you have made any modifications in steps 2-9, make sure to make the same modifications in step 11!

First, we read in the new, no-stars .json file with `with open('business_no_stars_review.json', encoding="utf8") as f:`. 

Then, we copy and paste the rest of the code from steps 2-9 on our current runme.py file to create a dataframe nearly identical to the one we created in step 9. The one difference between the code written in steps 2-9 and that written in step 11 is that we remove the code concerning the actual star count because this .json does not contain information about stars (that is, after all, what we're predicting.)

Again, if you've made modifications to the file in steps 2-9, make sure to also make those here because the trained models require that each variable present in the data for the model training is present for use in prediction.

By the end of step 11, you have a new dataset/DataFrame constructed with information from the no-stars .json file:
```
dataset = pandas.DataFrame(data={'name': data_name,
					'text_analysis_prediction': data_prediction,
					......
					'events': data_events,
					'nightlife': data_nightlife
						})
```

### Step 12: Make predictions

Finally, we use random forest to make predictions about the star count for the businesses present in the no-stars .json file (in this case, business_no_stars_review.json). 

First, we create an empty container for our predictions of star counts: `data_machine_prediction = []`.

Next, we restrict the dataset to nonrestaurant businesses with `dataset = dataset[(dataset['is_restaurant']!=1)]` in runme_nonrestaurants.py and restaurants with `dataset = dataset[(dataset['is_restaurant']==1)]` in runme_restaurants.py.
We now clean the dataset in the same way that we did earlier- by dropping rows with empty values. 
```
dataset = dataset.dropna()
dataset = dataset.replace(to_replace='None',value=np.nan).dropna()
```
Next, we show the program where to find the business's name in order to append it to a dataframe later in the program, and then we tell the program that want it to make its predictions on the basis of every column in the dataframe except for the business's name (which is a string, which doesn't work with random forest).
```
data_name = dataset['name']
data = dataset.iloc[:,1:].values
```
We then deserialize and access the trained random forest machine and use it to predict review ratings for the new busineses. 
```
file = open("nonrestaurants_serialized_random_forest_text.pickle", "rb")
# this opens the random forest that I trained earlier
machine = pickle.load(file)
# this accesses the RF machine I trained earlier.		
cell_machine_prediction = machine.predict(data)	
```
Having made predictions, we append these alongside the business names to a new dataframe.
```
df1 = pandas.DataFrame(data={'name': data_name,
	'predicted_stars': cell_machine_prediction})
```
We print the dataframe in the terminal so that users can conveniently view the model's predictions there.
```
print(df1)
```
We also save the dataframe as a csv so that users can access the predictions long after closing the terminal window.
```
df1.to_csv('nonrestaurants_predictions_with_text.csv')
```

### Step 13: Run the programs
Once you've read through and modified the code as outlined above, you are ready to run the two runme files: [runme_restaurants.py](https://github.com/cbouts/ECON498_Final/blob/master/runme_restaurants.py) and [runme_nonrestaurants.py](https://github.com/cbouts/ECON498_Final/blob/main/runme_nonrestaurants.py). This can be done simultaneously in two terminal windows. 

### See also report.pdf
I argue that my random forest models are good in [report.pdf](https://github.com/cbouts/ECON498_Final/blob/master/report.pdf)
