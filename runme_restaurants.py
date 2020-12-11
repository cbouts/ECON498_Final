import pandas
import json
import ast
import numpy as np
from statistics import mode
from statistics import mean
from datetime import datetime
import time
import pickle
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
# for text analysis:
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import string

# accounting for the restaurant's location:
data_name = []
data_stars = []
data_prediction = []
data_address = []
data_city = []
data_state = []
data_vegas = []
data_toronto = []
data_phoenix = []
data_clt = []
data_scottsdale = []
data_calgary = []
data_pitt = []
data_montreal = []
data_mesa = []
data_henderson = []
data_tempe = []
data_chandler = []
data_cleveland = []
data_glendale = []
data_madison = []
data_AB = []
data_AZ = []
data_IL = []
data_NC = []
data_NV = []
data_NY = []
data_OH = []
data_ON = []
data_PA = []
data_QC = []
data_SC = []
data_WI = []
data_CAN = []
data_UK = []
# don't include UK in data because it only has 11 observations!
data_US = []
data_latitude = []
data_longitude = []

# accounting for the restaurant's hours:
data_hours = []
data_monday = []
data_tuesday = []
data_wednesday = []
data_thursday = []
data_friday = []
data_saturday = []
data_sunday = []
data_weekends = []
data_monday_close = []
data_tuesday_close = []
data_wednesday_close = []
data_thursday_close = []
data_friday_close = []
data_saturday_close = []
data_sunday_close = []
data_monday_open = []
data_tuesday_open = []
data_wednesday_open = []
data_thursday_open = []
data_friday_open = []
data_saturday_open = []
data_sunday_open = []
data_days_open = []
data_open_mode = []
data_close_mode = []
data_monday_total_hours = []
data_tuesday_total_hours = []
data_wednesday_total_hours = []
data_thursday_total_hours = []
data_friday_total_hours = []
data_saturday_total_hours = []
data_sunday_total_hours = []
data_hours_open = []
data_mean_hours = []
data_total_hours = []

# recording the restaurant's attributes: 
list_attributes = []
# don't include "list_attributes" in the data.
data_outdoor_seating = []
data_credit_cards = []
data_take_out = []
data_groups = []
data_restaurant_price = []
data_caters = []
data_reservations = []
data_kids = []
data_tv = []
data_dinner = []
data_lunch = []
data_breakfast = []
data_latenight = []
data_dessert = []
data_alcohol = []
data_noise_quiet = []
data_noise_avg = []
data_noise_loud = []

# information about the restaurant's ambience:
data_touristy = []
data_classy = []
data_romantic = []
data_intimate = []
data_hipster = []
data_divey = []
data_trendy = []
data_upscale = []
data_casual = []

# information about parking: (only include parking_regular and valet in data)
data_garage = []
data_streetparking = []
data_validated = []
data_lot = []
data_valet = []
data_regular_parking = []

# whether or not it's a restaurant:
data_restaurant = []
# don't include "data_restaurant" in the data.

# accounting for different types of restaurant:
list_categories = []
# don't include "list_categories" in the data.
data_thai = []
data_malaysian = []
data_irish = []
data_steakhouses = []
data_burgers = []
data_fast_food = []
data_french = []
data_modern_european = []
data_seafood = []
data_mexican = []
data_bars = []
data_gastropubs = []
data_chinese = []
data_american_new = []
data_american_traditional = []
data_japanese = []
data_pizza = []
data_mediterranean = []
data_breakfast_brunch = []
data_delis = []
data_italian = []
data_diners = []
data_vegan = []
data_latin_american = []
data_german = []
data_coffee_tea = []
data_tapas = []
data_bakeries = []


with open('business_sample.json', encoding="utf8") as f:
	head = [next(f) for x in range(25000)]
	for line in head:
	# for line in f:
		json_line = json.loads(line)

		cell_name = json_line.get('name')
		data_name.append(str(cell_name))

		# # To get the stars score
		cell_stars = json_line.get('stars')
		data_stars.append(cell_stars)

		# add text analysis stuff here.
		cell_categories = json_line.get('categories')
		list_categories.append(cell_categories)

		if cell_categories is not None:
			cell_restaurant = (1 * ('Restaurants' in cell_categories))
			data_restaurant.append(cell_restaurant)
		else:
			data_restaurant.append(None)
		# text analysis

		if cell_restaurant == 1:

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
			# this appends an empty value for data_prediction for nonrestaurant businesses. 


		# # To get the address of the business
		cell_address = json_line.get('address')
		data_address.append(cell_address)


		cell_city = json_line.get('city')
		data_city.append(cell_city)

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

		# # To get the state
		cell_state = json_line.get('state')
		data_state.append(1 * cell_state)
		# # To get a variable indicating whether the business is in a given state (for states represented by over 900 businesses):
		if cell_state == "AB":
			data_AB.append(1)
		else:
			data_AB.append(0)
		if cell_state == "AZ":
			data_AZ.append(1)
		else:
			data_AZ.append(0)
		if cell_state == "IL":
			data_IL.append(1)
		else:
			data_IL.append(0)
		if cell_state == "NC":
			data_NC.append(1)
		else:
			data_NC.append(0)
		if cell_state == "NV":
			data_NV.append(1)
		else:
			data_NV.append(0)
		if cell_state == "NY":
			data_NY.append(1)
		else:
			data_NY.append(0)
		if cell_state == "OH":
			data_OH.append(1)
		else:
			data_OH.append(0)
		if cell_state == "ON":
			data_ON.append(1)
		else:
			data_ON.append(0)
		if cell_state == "PA":
			data_PA.append(1)
		else:
			data_PA.append(0)
		if cell_state == "QC":
			data_QC.append(1)
		else:
			data_QC.append(0)
		if cell_state == "SC":
			data_SC.append(1)
		else:
			data_SC.append(0)
		if cell_state == "WI":
			data_WI.append(1)
		else:
			data_WI.append(0)

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

		cell_latitude = json_line.get('latitude')
		data_latitude.append(cell_latitude)

		cell_longitude = json_line.get('longitude')
		data_longitude.append(cell_longitude)


# working with hours:
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

		# to see if the business is open at all on weekends:
		if cell_sunday is None and cell_saturday is None:
			data_weekends.append(0)
		else: 
			data_weekends.append(1)

		# getting closing and opening times (in seconds):
		if cell_monday is not None:
			cell_monday_close = pandas.to_timedelta((cell_monday.split('-')[1]) + ":00").total_seconds()
			if cell_monday_close == 0.0:
				cell_monday_close = 86399.0
			else:
				cell_monday_close = cell_monday_close
			cell_monday_open = pandas.to_timedelta((cell_monday.split('-')[0]) + ":00").total_seconds()
			if cell_monday_open == 0.0:
				cell_monday_open = 1.0
			else:
				cell_monday_open = cell_monday_open
		else:
			cell_monday_close = 0
			cell_monday_open = 0
		data_monday_close.append(cell_monday_close)
		data_monday_open.append(cell_monday_open)

		if cell_tuesday is not None:
			cell_tuesday_close = pandas.to_timedelta((cell_tuesday.split('-')[1]) + ":00").total_seconds()
			if cell_tuesday_close == 0.0:
				cell_tuesday_close = 86399.0
			else:
				cell_tuesday_close = cell_tuesday_close
			cell_tuesday_open = pandas.to_timedelta((cell_tuesday.split('-')[0]) + ":00").total_seconds()
			if cell_tuesday_open == 0.0:
				cell_tuesday_open = 1.0
			else:
				cell_tuesday_open = cell_tuesday_open
		else:
			cell_tuesday_close = 0
			cell_tuesday_open = 0
		data_tuesday_close.append(cell_tuesday_close)
		data_tuesday_open.append(cell_tuesday_open)

		if cell_wednesday is not None:
			cell_wednesday_close = pandas.to_timedelta((cell_wednesday.split('-')[1]) + ":00").total_seconds()
			if cell_wednesday_close == 0.0:
				cell_wednesday_close = 86399.0
			else:
				cell_wednesday_close = cell_wednesday_close
			cell_wednesday_open = pandas.to_timedelta((cell_wednesday.split('-')[0]) + ":00").total_seconds()
			if cell_wednesday_open == 0.0:
				cell_wednesday_open = 1.0
			else:
				cell_wednesday_open = cell_wednesday_open
		else:
			cell_wednesday_close = 0
			cell_wednesday_open = 0
		data_wednesday_close.append(cell_wednesday_close)
		data_wednesday_open.append(cell_wednesday_open)

		if cell_thursday is not None:
			cell_thursday_close = pandas.to_timedelta((cell_thursday.split('-')[1]) + ":00").total_seconds()
			if cell_thursday_close == 0.0:
				cell_thursday_close = 86399.0
			else:
				cell_thursday_close = cell_thursday_close
			cell_thursday_open = pandas.to_timedelta((cell_thursday.split('-')[0]) + ":00").total_seconds()
			if cell_thursday_open == 0.0:
				cell_thursday_open = 1.0
			else:
				cell_thursday_open = cell_thursday_open
		else:
			cell_thursday_close = 0
			cell_thursday_open = 0
		data_thursday_close.append(cell_thursday_close)
		data_thursday_open.append(cell_thursday_open)

		if cell_friday is not None:
			cell_friday_close = pandas.to_timedelta((cell_friday.split('-')[1]) + ":00").total_seconds()
			if cell_friday_close == 0.0:
				cell_friday_close = 86399.0
			else:
				cell_friday_close = cell_friday_close
			cell_friday_open = pandas.to_timedelta((cell_friday.split('-')[0]) + ":00").total_seconds()
			if cell_friday_open == 0.0:
				cell_friday_open = 1.0
			else:
				cell_friday_open = cell_friday_open
		else:
			cell_friday_close = 0
			cell_friday_open = 0
		data_friday_close.append(cell_friday_close)
		data_friday_open.append(cell_friday_open)

		if cell_saturday is not None:
			cell_saturday_close = pandas.to_timedelta((cell_saturday.split('-')[1]) + ":00").total_seconds()
			if cell_saturday_close == 0.0:
				cell_saturday_close = 86399.0
			else:
				cell_saturday_close = cell_saturday_close
			cell_saturday_open = pandas.to_timedelta((cell_saturday.split('-')[0]) + ":00").total_seconds()
			if cell_saturday_open == 0.0:
				cell_saturday_open = 1.0
			else:
				cell_saturday_open = cell_saturday_open
		else:
			cell_saturday_close = 0
			cell_saturday_open = 0
		data_saturday_close.append(cell_saturday_close)
		data_saturday_open.append(cell_saturday_open)

		if cell_sunday is not None:
			cell_sunday_close = pandas.to_timedelta((cell_sunday.split('-')[1]) + ":00").total_seconds()
			if cell_sunday_close == 0.0:
				cell_sunday_close = 86399.0
			else:
				cell_sunday_close = cell_sunday_close
			cell_sunday_open = pandas.to_timedelta((cell_sunday.split('-')[0]) + ":00").total_seconds()
			if cell_sunday_open == 0.0:
				cell_sunday_open = 1.0
			else:
				cell_sunday_open = cell_sunday_open
		else:
			cell_sunday_close = 0
			cell_sunday_open = 0
		data_sunday_close.append(cell_sunday_close)
		data_sunday_open.append(cell_sunday_open)

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


		# getting total hours per workday in seconds. it gives us negative float values, but this is fine as it preserves a relationship between rating stars and total hours open.
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


		if cell_tuesday is not None:
			if cell_tuesday_open < cell_tuesday_close:
				cell_tuesday_total_hours = (cell_tuesday_open - cell_tuesday_close)
			elif cell_tuesday_open > cell_tuesday_close:
				cell_tuesday_total_hours = (cell_tuesday_close - cell_tuesday_open)
			else:
				cell_tuesday_total_hours = 0
		else:
			cell_tuesday_total_hours = 0
		data_tuesday_total_hours.append(cell_tuesday_total_hours)

		if cell_wednesday is not None:
			if cell_wednesday_open < cell_wednesday_close:
				cell_wednesday_total_hours = (cell_wednesday_open - cell_wednesday_close)
			elif cell_wednesday_open > cell_wednesday_close:
				cell_wednesday_total_hours = (cell_wednesday_close - cell_wednesday_open)
			else:
				cell_wednesday_total_hours = 0
		else:
			cell_wednesday_total_hours = 0
		data_wednesday_total_hours.append(cell_wednesday_total_hours)

		if cell_thursday is not None:
			if cell_thursday_open < cell_thursday_close:
				cell_thursday_total_hours = (cell_thursday_open - cell_thursday_close)
			elif cell_thursday_open > cell_thursday_close:
				cell_thursday_total_hours = (cell_thursday_close - cell_thursday_open)
			else:
				cell_thursday_total_hours = 0
		else:
			cell_thursday_total_hours = 0
		data_thursday_total_hours.append(cell_thursday_total_hours)

		if cell_friday is not None:
			if cell_friday_open < cell_friday_close:
				cell_friday_total_hours = (cell_friday_open - cell_friday_close)
			elif cell_friday_open > cell_friday_close:
				cell_friday_total_hours = (cell_friday_close - cell_friday_open)
			else:
				cell_friday_total_hours = 0
		else:
			cell_friday_total_hours = 0
		data_friday_total_hours.append(cell_friday_total_hours)

		if cell_saturday is not None:
			if cell_saturday_open < cell_saturday_close:
				cell_saturday_total_hours = (cell_saturday_open - cell_saturday_close)
			elif cell_saturday_open > cell_saturday_close:
				cell_saturday_total_hours = (cell_saturday_close - cell_saturday_open)
			else:
				cell_saturday_total_hours = 0
		else:
			cell_saturday_total_hours = 0
		data_saturday_total_hours.append(cell_saturday_total_hours)


		if cell_sunday is not None:
			if cell_sunday_open < cell_sunday_close:
				cell_sunday_total_hours = (cell_sunday_open - cell_sunday_close)
			elif cell_sunday_open > cell_sunday_close:
				cell_sunday_total_hours = (cell_sunday_close - cell_sunday_open)
			else:
				cell_sunday_total_hours = 0
		else:
			cell_sunday_total_hours = 0
		data_sunday_total_hours.append(cell_sunday_total_hours)


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

# to see how  many days a business is open each week:
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

# the values given are negative, but that's okay because it preserves the relationship between stars and hours



		cell_attribute = json_line.get('attributes')
		# list_attributes.append(1 * cell_attribute)

		if cell_attribute is not None:
			cell_outdoor_seating = cell_attribute.get('OutdoorSeating')
			if cell_outdoor_seating is not None:
				if cell_outdoor_seating == True:
					data_outdoor_seating.append(1)
				else:
					data_outdoor_seating.append(0)
			else:
				data_outdoor_seating.append(0)
		else:
			data_outdoor_seating.append(None)

		# # To get whether the business is by appointment only, 'None' if the business did not specified that of it does not have any attributes.
		if cell_attribute is not None:
			cell_credit_cards = cell_attribute.get('BusinessAcceptsCreditCards')
			if cell_credit_cards is not None:
				if cell_credit_cards == True:
					data_credit_cards.append(1)
				else:
					data_credit_cards.append(0)
			else:
				data_credit_cards.append(0)
		else:
			data_credit_cards.append(None)

		if cell_attribute is not None:
			cell_take_out = cell_attribute.get('RestaurantsTakeOut')
			if cell_take_out is not None:
				if cell_take_out == True:
					data_take_out.append(1)
				else:
					data_take_out.append(0)
			else:
				data_take_out.append(0)
		else:
			data_take_out.append(None)


		if cell_attribute is not None:
			cell_groups = cell_attribute.get('RestaurantsGoodForGroups')
			if cell_groups is not None:
				if cell_groups == True:
					data_groups.append(1)
				else:
					data_groups.append(0)
			else:
				data_groups.append(0)
		else:
			data_groups.append(None)


		# # To get restaurant price range, 'None' if the business did not specified that or it does not have any attributes.
		if cell_attribute is not None:
			cell_restaurant_price = cell_attribute.get('RestaurantsPriceRange2')
			if cell_restaurant_price is not None:
				data_restaurant_price.append(cell_restaurant_price)
			else:
				data_restaurant_price.append(0)
		else:
			data_restaurant_price.append(None)

		if cell_attribute is not None:
			cell_caters = cell_attribute.get('Caters')
			if cell_caters is not None:
				if cell_caters == True:
					data_caters.append(1)
				else:
					data_caters.append(0)
			else:
				data_caters.append(0)
		else:
			data_caters.append(None)

		if cell_attribute is not None:
			cell_reservations = cell_attribute.get('RestaurantsReservations')
			if cell_reservations is not None:
				if cell_reservations == True:
					data_reservations.append(1)
				else:
					data_reservations.append(0)
			else:
				data_reservations.append(0)
		else:
			data_reservations.append(None)

		if cell_attribute is not None:
			cell_kids = cell_attribute.get('GoodForKids')
			if cell_kids is not None:
				if cell_kids == True:
					data_kids.append(1)
				else:
					data_kids.append(0)
			else:
				data_kids.append(0)
		else:
			data_kids.append(None)

		# if cell_attribute is not None:
		# 	cell_tv = cell_attribute.get('HasTV')
		# 	if cell_tv is not None:
		# 		if cell_kids == True:
		# 			data_kids.append(1)
		# 		else:
		# 			data_kids.append(0)
		# 	else:
		# 		data_tv.append(0)
		# else:
		# 	data_tv.append(None)

		if cell_attribute is not None:
			cell_meal = cell_attribute.get('GoodForMeal')
			if cell_meal is not None:
				x = ast.literal_eval(cell_meal)
				if x is not None:
					try:
						cell_dinner = x['dinner']
						if cell_dinner is not None:
							data_dinner.append(1 * cell_dinner)
						else:
							data_dinner.append(0)
					except KeyError:
						data_dinner.append(None)

					try:
						cell_lunch = x['lunch']
						if cell_lunch is not None:
							data_lunch.append(1 * cell_lunch)
						else:
							data_lunch.append(0)
					except KeyError:
						data_lunch.append(None)

					try:
						cell_breakfast = x['breakfast']
						if cell_breakfast is not None:
							data_breakfast.append(1 * cell_breakfast)
						else:
							data_breakfast.append(0)
					except KeyError:
						data_breakfast.append(None)

					try:
						cell_latenight = x['latenight']
						if cell_latenight is not None:
							data_latenight.append(1 * cell_latenight)
						else:
							data_latenight.append(0)
					except KeyError:
						data_latenight.append(None)

					try:
						cell_dessert = x['dessert']
						if cell_dessert is not None:
							data_dessert.append(1 * cell_dessert)
						else:
							data_dessert.append(0)
					except KeyError:
						data_dessert.append(None)

				else:
					data_dinner.append(None)
					data_lunch.append(None)
					data_breakfast.append(None)
					data_latenight.append(None)
					data_dessert.append(None)
			else:
				data_dinner.append(None)
				data_lunch.append(None)
				data_breakfast.append(None)
				data_latenight.append(None)
				data_dessert.append(None)
		else:
			data_dinner.append(None)
			data_lunch.append(None)
			data_breakfast.append(None)
			data_latenight.append(None)
			data_dessert.append(None)


		if cell_attribute is not None:
			cell_alcohol = cell_attribute.get('Alcohol')
			if cell_alcohol is not None:
				if cell_alcohol != "u'none'" and cell_alcohol != "'none'":
					data_alcohol.append(1)
				else:
					data_alcohol.append(0)
			else:
				data_alcohol.append(0)
		else:
			data_alcohol.append(None)

		if cell_attribute is not None:
			cell_noise_level = cell_attribute.get('NoiseLevel')
			if cell_noise_level is not None:
				if cell_noise_level == "u'quiet'" or cell_noise_level == "'quiet'":
					data_noise_quiet.append(1)
				else:
					data_noise_quiet.append(0)
			else:
				data_noise_quiet.append(0)
		else:
			data_noise_quiet.append(None)

		if cell_attribute is not None:
			cell_noise_level = cell_attribute.get('NoiseLevel')
			if cell_noise_level is not None:
				if cell_noise_level == "u'average'" or cell_noise_level == "'average'":
					data_noise_avg.append(1)
				else:
					data_noise_avg.append(0)
			else:
				data_noise_avg.append(0)
		else:
			data_noise_avg.append(None)

		if cell_attribute is not None:
			cell_noise_level = cell_attribute.get('NoiseLevel')
			if cell_noise_level is not None:
				if cell_noise_level == "u'loud'" or cell_noise_level == "'loud'":
					data_noise_loud.append(1)
				else:
					data_noise_loud.append(0)
			else:
				data_noise_loud.append(0)
		else:
			data_noise_loud.append(None)

		if cell_attribute is not None:
			cell_ambience = cell_attribute.get('Ambience')
			if cell_ambience is not None:
				x = ast.literal_eval(cell_ambience)
				if x is not None:
					try:
						cell_touristy = x['touristy']
						if cell_touristy is not None:
							data_touristy.append(1 * cell_touristy)
						else:
							data_touristy.append(0)
					except KeyError:
						data_touristy.append(None)

					try:
						cell_classy = x['classy']
						if cell_classy is not None:
							data_classy.append(1 * cell_classy)
						else:
							data_classy.append(0)
					except KeyError:
						data_classy.append(None)

					try:
						cell_romantic = x['romantic']
						if cell_romantic is not None:
							data_romantic.append(1 * cell_romantic)
						else:
							data_romantic.append(0)
					except KeyError:
						data_romantic.append(None)

					try:
						cell_intimate = x['intimate']
						if cell_intimate is not None:
							data_intimate.append(1 * cell_intimate)
						else:
							data_intimate.append(0)
					except KeyError:
						data_intimate.append(None)

					try:
						cell_hipster = x['hipster']
						if cell_hipster is not None:
							data_hipster.append(1 * cell_hipster)
						else:
							data_hipster.append(0)
					except KeyError:
						data_hipster.append(None)

					try:
						cell_divey = x['divey']
						if cell_divey is not None:
							data_divey.append(1 * cell_divey)
						else:
							data_divey.append(0)
					except KeyError:
						data_divey.append(None)

					try:
						cell_trendy = x['trendy']
						if cell_trendy is not None:
							data_trendy.append(1 * cell_trendy)
						else:
							data_trendy.append(0)
					except KeyError:
						data_trendy.append(None)

					try:
						cell_upscale = x['upscale']
						if cell_upscale is not None:
							data_upscale.append(1 * cell_upscale)
						else:
							data_upscale.append(0)
					except KeyError:
						data_upscale.append(None)

					try:
						cell_casual = x['casual']
						if cell_casual is not None:
							data_casual.append(1 * cell_casual)
						else:
							data_casual.append(0)
					except KeyError:
						data_casual.append(None)

				else:
					data_touristy.append(None)
					data_classy.append(None)
					data_romantic.append(None)
					data_intimate.append(None)
					data_hipster.append(None)
					data_divey.append(None)
					data_trendy.append(None)
					data_upscale.append(None)
					data_casual.append(None)
			else:
				data_touristy.append(None)
				data_classy.append(None)
				data_romantic.append(None)
				data_intimate.append(None)
				data_hipster.append(None)
				data_divey.append(None)
				data_trendy.append(None)
				data_upscale.append(None)
				data_casual.append(None)
		else:
			data_touristy.append(None)
			data_classy.append(None)
			data_romantic.append(None)
			data_intimate.append(None)
			data_hipster.append(None)
			data_divey.append(None)
			data_trendy.append(None)
			data_upscale.append(None)
			data_casual.append(None)


		if cell_attribute is not None:
			cell_parking = cell_attribute.get('BusinessParking')
			if cell_parking is not None:
				x = ast.literal_eval(cell_parking)
				if x is not None:
					try:
						cell_garage = x['garage']
						if cell_garage is not None:
							data_garage.append(1 * cell_garage)
						else:
							data_garage.append(0)
					except KeyError:
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
					data_garage.append(None)
					data_streetparking.append(None)
					data_validated.append(None)
					data_lot.append(None)
					data_valet.append(None)
					data_regular_parking.append(None)
			else:
				data_garage.append(None)
				data_streetparking.append(None)
				data_validated.append(None)
				data_lot.append(None)
				data_valet.append(None)
				data_regular_parking.append(None)
		else:
			data_garage.append(None)
			data_streetparking.append(None)
			data_validated.append(None)
			data_lot.append(None)
			data_valet.append(None)
			data_regular_parking.append(None)


		# # accounting for types of restaurants:
		if cell_categories is not None:
			cell_thai = ('Thai' in cell_categories)
			data_thai.append(1 * cell_thai)
		else:
			data_thai.append(None)

		if cell_categories is not None:
			cell_malaysian = ('Malaysian' in cell_categories)
			data_malaysian.append(1 * cell_malaysian)
		else:
			data_malaysian.append(None)

		if cell_categories is not None:
			cell_irish = ('Irish' in cell_categories)
			data_irish.append(1 * cell_irish)
		else:
			data_irish.append(None)

		if cell_categories is not None:
			cell_steakhouses = ('Steakhouses' in cell_categories)
			data_steakhouses.append(1 * cell_steakhouses)
		else:
			data_steakhouses.append(None)

		if cell_categories is not None:
			cell_burgers = ('Burgers' in cell_categories)
			data_burgers.append(1 * cell_burgers)
		else:
			data_burgers.append(None)

		if cell_categories is not None:
			cell_fast_food = ('Fast Food' in cell_categories)
			data_fast_food.append(1 * cell_fast_food)
		else:
			data_fast_food.append(None)

		if cell_categories is not None:
			cell_french = ('French' in cell_categories)
			data_french.append(1 * cell_french)
		else:
			data_french.append(None)

		if cell_categories is not None:
			cell_modern_european = ('Modern European' in cell_categories)
			data_modern_european.append(1 * cell_modern_european)
		else:
			data_modern_european.append(None)		

		if cell_categories is not None:
			cell_seafood = ('Seafood' in cell_categories)
			data_seafood.append(1 * cell_seafood)
		else:
			data_seafood.append(None)

		if cell_categories is not None:
			cell_mexican = ('Mexican' in cell_categories)
			data_mexican.append(1 * cell_mexican)
		else:
			data_mexican.append(None)

		if cell_categories is not None:
			cell_bars = ('Bars' in cell_categories)
			data_bars.append(1 * cell_bars)
		else:
			data_bars.append(None)

		if cell_categories is not None:
			cell_gastropubs = ('Gastropubs' in cell_categories)
			data_gastropubs.append(1 * cell_gastropubs)
		else:
			data_gastropubs.append(None)

		if cell_categories is not None:
			cell_chinese = ('Chinese' in cell_categories)
			data_chinese.append(1 * cell_chinese)
		else:
			data_chinese.append(None)

		if cell_categories is not None:
			cell_data_american_new = ('American (New)' in cell_categories)
			data_american_new.append(1 * cell_data_american_new)
		else:
			data_american_new.append(None)

		if cell_categories is not None:
			cell_american_traditional = ('American (Traditional)' in cell_categories)
			data_american_traditional.append(1 * cell_american_traditional)
		else:
			data_american_traditional.append(None)

		if cell_categories is not None:
			cell_japanese = ('Japanese' in cell_categories)
			data_japanese.append(1 * cell_japanese)
		else:
			data_japanese.append(None)

		if cell_categories is not None:
			cell_pizza = ('Pizza' in cell_categories)
			data_pizza.append(1 * cell_pizza)
		else:
			data_pizza.append(None)

		if cell_categories is not None:
			cell_mediterranean = ('Mediterranean' in cell_categories)
			data_mediterranean.append(1 * cell_mediterranean)
		else:
			data_mediterranean.append(None)

		if cell_categories is not None:
			cell_breakfast_brunch = ('Breakfast & Brunch' in cell_categories)
			data_breakfast_brunch.append(1 * cell_breakfast_brunch)
		else:
			data_breakfast_brunch.append(None)

		if cell_categories is not None:
			cell_delis = ('Delis' in cell_categories)
			data_delis.append(1 * cell_delis)
		else:
			data_delis.append(None)

		if cell_categories is not None:
			cell_italian = ('Italian' in cell_categories)
			data_italian.append(1 * cell_italian)
		else:
			data_italian.append(None)

		if cell_categories is not None:
			cell_diners = ('Diners' in cell_categories)
			data_diners.append(1 * cell_diners)
		else:
			data_diners.append(None)

		if cell_categories is not None:
			cell_vegan = ('Vegan' in cell_categories)
			data_vegan.append(1 * cell_vegan)
		else:
			data_vegan.append(None)

		if cell_categories is not None:
			cell_latin_american = ('Latin American' in cell_categories)
			data_latin_american.append(1 * cell_latin_american)
		else:
			data_latin_american.append(None)

		if cell_categories is not None:
			cell_german = ('German' in cell_categories)
			data_german.append(1 * cell_german)
		else:
			data_german.append(None)

		if cell_categories is not None:
			cell_coffee_tea = ('Coffee & Tea' in cell_categories)
			data_coffee_tea.append(1 * cell_coffee_tea)
		else:
			data_coffee_tea.append(None)

		if cell_categories is not None:
			cell_tapas = ('Tapas\/Small Plates' in cell_categories)
			data_tapas.append(1 * cell_tapas)
		else:
			data_tapas.append(None)

		if cell_categories is not None:
			cell_bakeries = ('Bakeries' in cell_categories)
			data_bakeries.append(1 * cell_bakeries)
		else:
			data_bakeries.append(None)


dataset = pandas.DataFrame(data={'name': data_name,
					'stars': data_stars,
					'text_analysis_prediction': data_prediction,
					# 'address': data_address,
					# 'state': data_state,
					# 'city': data_city,
					'Las_Vegas': data_vegas, 
					'Toronto': data_toronto, 
					'Phoenix': data_phoenix, 
					'Charlotte': data_clt, 
					'Scottsdale': data_scottsdale, 
					'Calgary': data_calgary, 
					'Pittsburgh': data_pitt, 
					'Montréal': data_montreal, 
					'Mesa': data_mesa, 
					'Henderson': data_henderson, 
					'Tempe': data_tempe,
					'Chandler': data_chandler, 
					'Cleveland': data_cleveland, 
					'Glendale': data_glendale, 
					'Madison': data_madison,
					'AB': data_AB,
					'AZ': data_AZ,
					'IL': data_IL,
					'NC': data_NC,
					'NV': data_NV,
					'NY': data_NY,
					'OH': data_OH,
					'ON': data_ON,
					'PA': data_PA,
					'QC': data_QC,
					'SC': data_SC,
					'WI': data_WI,
					'CAN': data_CAN,
					'UK': data_UK,
					'US': data_US,
					'latitude': data_latitude,
					'longitude': data_longitude,
					# 'hours': data_hours,
					# 'mon': data_monday,
					# 'tues': data_tuesday,
					# 'weds': data_wednesday,
					# 'thurs': data_thursday,
					# 'fri': data_friday,
					# 'sat': data_saturday,
					# 'sun': data_sunday,
					'open_on_weekends': data_weekends,
					'mon_close': data_monday_close,
					'tues_close': data_tuesday_close,
					'weds_close': data_wednesday_close,
					'thurs_close': data_thursday_close,
					'fri_close': data_friday_close,
					'sat_close': data_saturday_close,
					'sun_close': data_sunday_close,
					'mon_open': data_monday_open,
					'tues_open': data_tuesday_open,
					'weds_open': data_wednesday_open,
					'thurs_open': data_thursday_open,
					'fri_open': data_friday_open,
					'sat_open': data_saturday_open,
					'sun_open': data_sunday_open,
					'monday_total_hours': data_monday_total_hours,
					'tuesday_total_hours': data_tuesday_total_hours,
					'wednesday_total_hours': data_wednesday_total_hours,
					'thursday_total_hours': data_thursday_total_hours,
					'friday_total_hours': data_friday_total_hours,
					'saturday_total_hours': data_saturday_total_hours,
					'sunday_total_hours': data_sunday_total_hours,
					'days_open': data_days_open,
					'mode_open_time': data_open_mode,
					'mode_close_time': data_close_mode,
					'mean_hours_per_day': data_mean_hours,
					'hours_per_week': data_total_hours,
					# 'attributes_list': list_attributes, 13^
					'outdoor_seating': data_outdoor_seating,
					'credit_cards': data_credit_cards,
					'take_out': data_take_out,
					'groups': data_groups,
					'restaurant_price': data_restaurant_price,
					# # couldn't get rest price to load as an integer.
					'caters': data_caters,
					'reservations': data_reservations,
					'kids': data_kids,
					# 'has_tv': data_tv,
					# didn't include has tv
					'dinner': data_dinner,
					'lunch': data_lunch,
					'breakfast': data_breakfast,
					'latenight': data_latenight,
					'dessert': data_dessert,
					'has_alcohol': data_alcohol,
					'noise_quiet': data_noise_quiet,
					'noise_avg': data_noise_avg,
					'noise_loud': data_noise_loud,
					'touristy': data_touristy,
					'classy': data_classy,
					'romantic': data_romantic,
					'intimate': data_intimate,
					'hipster': data_hipster,
					'divey': data_divey,
					'trendy': data_trendy,
					'upscale': data_upscale,
					'casual': data_casual,
					'garage': data_garage,
					'street_parking': data_streetparking,
					'validated': data_validated,
					'lot': data_lot,
					'valet': data_valet,
					'regular_parking': data_regular_parking,
					'is_restaurant': data_restaurant,
					# 'categories_list': list_categories,
					'thai': data_thai,
					'malaysian': data_malaysian,
					'irish': data_irish,
					'steakhouses': data_steakhouses,
					'burgers': data_burgers,
					'fast_food': data_fast_food,
					'data_french': data_french,
					'modern_european': data_modern_european,
					'seafood': data_seafood,
					'mexican': data_mexican,
					'bars': data_bars,
					'gastropubs': data_gastropubs,
					'chinese': data_chinese,
					'american_new': data_american_new,
					'american_traditional': data_american_traditional,
					'japanese': data_japanese,
					'pizza': data_pizza,
					'mediterranean': data_mediterranean,
					'breakfast_brunch': data_breakfast_brunch,
					'delis': data_delis,
					'italian': data_italian,
					'diners': data_diners,
					'vegan': data_vegan,
					'latin_america': data_latin_american,
					'german': data_german,
					'coffee_tea': data_coffee_tea,
					'tapas': data_tapas,
					'bakeries': data_bakeries
						})

dataset = dataset[(dataset['is_restaurant']==1)]
dataset = dataset.dropna()
dataset = dataset.replace(to_replace='None',value=np.nan).dropna()


print("dataframe created")




# train the machine using 
dataset = dataset[(dataset['stars']==1)|(dataset['stars']==2)|(dataset['stars']==3)|(dataset['stars']==4)|(dataset['stars']==5)]

# training our random forest model:
target = dataset.iloc[:,1].values
# this sets stars as the target.
data = dataset.iloc[:,2:].values

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size=0.25)

machine = RandomForestClassifier(n_estimators=250, criterion="gini", max_depth=23)
# 200 and 25 gave accuracy score .68
# 500 and 25 gave .67
# 500 and 15 gave .66
# 200 and 35 gave .717
# 200 and 40 gave .673
# 500 and 35 gave .68
# 200 and 35 gave .714
# 300 and 35 gave .706
# 250 and 35 gave .722
# 250 and 36 gave .68
# 250 and 37 gave .7
# 300 and 37 gave .68
# 300 and 36 gave .71
# 1000 and 36 gave .69
# 260 and 36 gave .69
# 250 and 36 gave .703
# 250 and 23 gave .708
# 1250 and 23 gave .716
# 650 and 25 gave .711
# 250 and 28 gave .666
# 250 and 26 gave .708
# 250 and 24 gave .738
# 300 and 24 gave .71
# 250 and 23 gave .740
# 600 and 25 gave .71
# so we decide to use 250 estimators and max depth 23





machine.fit(data_training, target_training)

prediction = machine.predict(data_test)
print("Confusion matrix from restaurant random forest:")
print(confusion_matrix(target_test, prediction))

# # print(machine)
print("Accuracy score from restaurant random forest:")
print(accuracy_score(target_test, prediction))

with open("restaurants_serialized_random_forest_text.pickle", "wb") as file:
	pickle.dump(machine, file)
# 	# dump() serializes to an open file (file-like object), dumps() serializes to a string.


# linear for comparison:

machine = linear_model.LinearRegression()
machine.fit(data_training, target_training)
with open("restaurants_serialized_linear_text.pickle", "wb") as file:
	# formerly "restaurants_serialized_lineart.pickle"
	pickle.dump(machine, file)

prediction = machine.predict(data_test)
print("............")
print("r2 score from restaurant linear model")
print(r2_score(target_test, prediction))


# ___________________________________________________________________________
# PART B: PREDICTING STARS USING OUR TRAINED MODEL.

# to do this, we copy and paste the file-reading/dataframe construction sections of our earlier code, remove infomration about "stars" from the code (since the .json file doesn't include stars), and change the name of the file being read to the file containing businesses we wish to predict stars for.


# accounting for the restaurant's location:
data_name = []
data_prediction = []
data_address = []
data_city = []
data_state = []
data_vegas = []
data_toronto = []
data_phoenix = []
data_clt = []
data_scottsdale = []
data_calgary = []
data_pitt = []
data_montreal = []
data_mesa = []
data_henderson = []
data_tempe = []
data_chandler = []
data_cleveland = []
data_glendale = []
data_madison = []
data_AB = []
data_AZ = []
data_IL = []
data_NC = []
data_NV = []
data_NY = []
data_OH = []
data_ON = []
data_PA = []
data_QC = []
data_SC = []
data_WI = []
data_CAN = []
data_UK = []
# don't include UK in data because it only has 11 observations!
data_US = []
data_latitude = []
data_longitude = []

# accounting for the restaurant's hours:
data_hours = []
data_monday = []
data_tuesday = []
data_wednesday = []
data_thursday = []
data_friday = []
data_saturday = []
data_sunday = []
data_weekends = []
data_monday_close = []
data_tuesday_close = []
data_wednesday_close = []
data_thursday_close = []
data_friday_close = []
data_saturday_close = []
data_sunday_close = []
data_monday_open = []
data_tuesday_open = []
data_wednesday_open = []
data_thursday_open = []
data_friday_open = []
data_saturday_open = []
data_sunday_open = []
data_days_open = []
data_open_mode = []
data_close_mode = []
data_monday_total_hours = []
data_tuesday_total_hours = []
data_wednesday_total_hours = []
data_thursday_total_hours = []
data_friday_total_hours = []
data_saturday_total_hours = []
data_sunday_total_hours = []
data_hours_open = []
data_mean_hours = []
data_total_hours = []

# recording the restaurant's attributes: 
list_attributes = []
# don't include "list_attributes" in the data.
data_outdoor_seating = []
data_credit_cards = []
data_take_out = []
data_groups = []
data_restaurant_price = []
data_caters = []
data_reservations = []
data_kids = []
data_tv = []
data_dinner = []
data_lunch = []
data_breakfast = []
data_latenight = []
data_dessert = []
data_alcohol = []
data_noise_quiet = []
data_noise_avg = []
data_noise_loud = []

# information about the restaurant's ambience:
data_touristy = []
data_classy = []
data_romantic = []
data_intimate = []
data_hipster = []
data_divey = []
data_trendy = []
data_upscale = []
data_casual = []

# information about parking:
data_garage = []
data_streetparking = []
data_validated = []
data_lot = []
data_valet = []
data_regular_parking = []

# whether or not it's a restaurant:
data_restaurant = []

# accounting for different types of restaurant:
list_categories = []
data_thai = []
data_malaysian = []
data_irish = []
data_steakhouses = []
data_burgers = []
data_fast_food = []
data_french = []
data_modern_european = []
data_seafood = []
data_mexican = []
data_bars = []
data_gastropubs = []
data_chinese = []
data_american_new = []
data_american_traditional = []
data_japanese = []
data_pizza = []
data_mediterranean = []
data_breakfast_brunch = []
data_delis = []
data_italian = []
data_diners = []
data_vegan = []
data_latin_american = []
data_german = []
data_coffee_tea = []
data_tapas = []
data_bakeries = []


with open('business_no_stars_review.json', encoding="utf8") as f:
	for line in f:
		json_line = json.loads(line)

		cell_name = json_line.get('name')
		data_name.append(str(cell_name))

		# # We don't include stars here because our new dataset doesn't include them

		# add text analysis stuff here.
		cell_categories = json_line.get('categories')
		list_categories.append(cell_categories)

		if cell_categories is not None:
			cell_restaurant = (1 * ('Restaurants' in cell_categories))
			data_restaurant.append(cell_restaurant)
		else:
			data_restaurant.append(None)
		# text analysis

		if cell_restaurant == 1:

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

			# print(deserialized_object)

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
			# this appends an empty value for data_prediction for nonrestaurant businesses. 


		# # To get the address of the business
		cell_address = json_line.get('address')
		data_address.append(cell_address)


		cell_city = json_line.get('city')
		data_city.append(cell_city)

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

		# # To get the state
		cell_state = json_line.get('state')
		data_state.append(1 * cell_state)
		# # To get a variable indicating whether the business is in a given state (for states represented by over 900 businesses):
		if cell_state == "AB":
			data_AB.append(1)
		else:
			data_AB.append(0)
		if cell_state == "AZ":
			data_AZ.append(1)
		else:
			data_AZ.append(0)
		if cell_state == "IL":
			data_IL.append(1)
		else:
			data_IL.append(0)
		if cell_state == "NC":
			data_NC.append(1)
		else:
			data_NC.append(0)
		if cell_state == "NV":
			data_NV.append(1)
		else:
			data_NV.append(0)
		if cell_state == "NY":
			data_NY.append(1)
		else:
			data_NY.append(0)
		if cell_state == "OH":
			data_OH.append(1)
		else:
			data_OH.append(0)
		if cell_state == "ON":
			data_ON.append(1)
		else:
			data_ON.append(0)
		if cell_state == "PA":
			data_PA.append(1)
		else:
			data_PA.append(0)
		if cell_state == "QC":
			data_QC.append(1)
		else:
			data_QC.append(0)
		if cell_state == "SC":
			data_SC.append(1)
		else:
			data_SC.append(0)
		if cell_state == "WI":
			data_WI.append(1)
		else:
			data_WI.append(0)

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

		cell_latitude = json_line.get('latitude')
		data_latitude.append(cell_latitude)

		cell_longitude = json_line.get('longitude')
		data_longitude.append(cell_longitude)


# working with hours:
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

		# to see if the business is open at all on weekends:
		if cell_sunday is None and cell_saturday is None:
			data_weekends.append(0)
		else: 
			data_weekends.append(1)

		# getting closing and opening times (in seconds):
		if cell_monday is not None:
			cell_monday_close = pandas.to_timedelta((cell_monday.split('-')[1]) + ":00").total_seconds()
			if cell_monday_close == 0.0:
				cell_monday_close = 86399.0
			else:
				cell_monday_close = cell_monday_close
			cell_monday_open = pandas.to_timedelta((cell_monday.split('-')[0]) + ":00").total_seconds()
			if cell_monday_open == 0.0:
				cell_monday_open = 1.0
			else:
				cell_monday_open = cell_monday_open
		else:
			cell_monday_close = 0
			cell_monday_open = 0
		data_monday_close.append(cell_monday_close)
		data_monday_open.append(cell_monday_open)

		if cell_tuesday is not None:
			cell_tuesday_close = pandas.to_timedelta((cell_tuesday.split('-')[1]) + ":00").total_seconds()
			if cell_tuesday_close == 0.0:
				cell_tuesday_close = 86399.0
			else:
				cell_tuesday_close = cell_tuesday_close
			cell_tuesday_open = pandas.to_timedelta((cell_tuesday.split('-')[0]) + ":00").total_seconds()
			if cell_tuesday_open == 0.0:
				cell_tuesday_open = 1.0
			else:
				cell_tuesday_open = cell_tuesday_open
		else:
			cell_tuesday_close = 0
			cell_tuesday_open = 0
		data_tuesday_close.append(cell_tuesday_close)
		data_tuesday_open.append(cell_tuesday_open)

		if cell_wednesday is not None:
			cell_wednesday_close = pandas.to_timedelta((cell_wednesday.split('-')[1]) + ":00").total_seconds()
			if cell_wednesday_close == 0.0:
				cell_wednesday_close = 86399.0
			else:
				cell_wednesday_close = cell_wednesday_close
			cell_wednesday_open = pandas.to_timedelta((cell_wednesday.split('-')[0]) + ":00").total_seconds()
			if cell_wednesday_open == 0.0:
				cell_wednesday_open = 1.0
			else:
				cell_wednesday_open = cell_wednesday_open
		else:
			cell_wednesday_close = 0
			cell_wednesday_open = 0
		data_wednesday_close.append(cell_wednesday_close)
		data_wednesday_open.append(cell_wednesday_open)

		if cell_thursday is not None:
			cell_thursday_close = pandas.to_timedelta((cell_thursday.split('-')[1]) + ":00").total_seconds()
			if cell_thursday_close == 0.0:
				cell_thursday_close = 86399.0
			else:
				cell_thursday_close = cell_thursday_close
			cell_thursday_open = pandas.to_timedelta((cell_thursday.split('-')[0]) + ":00").total_seconds()
			if cell_thursday_open == 0.0:
				cell_thursday_open = 1.0
			else:
				cell_thursday_open = cell_thursday_open
		else:
			cell_thursday_close = 0
			cell_thursday_open = 0
		data_thursday_close.append(cell_thursday_close)
		data_thursday_open.append(cell_thursday_open)

		if cell_friday is not None:
			cell_friday_close = pandas.to_timedelta((cell_friday.split('-')[1]) + ":00").total_seconds()
			if cell_friday_close == 0.0:
				cell_friday_close = 86399.0
			else:
				cell_friday_close = cell_friday_close
			cell_friday_open = pandas.to_timedelta((cell_friday.split('-')[0]) + ":00").total_seconds()
			if cell_friday_open == 0.0:
				cell_friday_open = 1.0
			else:
				cell_friday_open = cell_friday_open
		else:
			cell_friday_close = 0
			cell_friday_open = 0
		data_friday_close.append(cell_friday_close)
		data_friday_open.append(cell_friday_open)

		if cell_saturday is not None:
			cell_saturday_close = pandas.to_timedelta((cell_saturday.split('-')[1]) + ":00").total_seconds()
			if cell_saturday_close == 0.0:
				cell_saturday_close = 86399.0
			else:
				cell_saturday_close = cell_saturday_close
			cell_saturday_open = pandas.to_timedelta((cell_saturday.split('-')[0]) + ":00").total_seconds()
			if cell_saturday_open == 0.0:
				cell_saturday_open = 1.0
			else:
				cell_saturday_open = cell_saturday_open
		else:
			cell_saturday_close = 0
			cell_saturday_open = 0
		data_saturday_close.append(cell_saturday_close)
		data_saturday_open.append(cell_saturday_open)

		if cell_sunday is not None:
			cell_sunday_close = pandas.to_timedelta((cell_sunday.split('-')[1]) + ":00").total_seconds()
			if cell_sunday_close == 0.0:
				cell_sunday_close = 86399.0
			else:
				cell_sunday_close = cell_sunday_close
			cell_sunday_open = pandas.to_timedelta((cell_sunday.split('-')[0]) + ":00").total_seconds()
			if cell_sunday_open == 0.0:
				cell_sunday_open = 1.0
			else:
				cell_sunday_open = cell_sunday_open
		else:
			cell_sunday_close = 0
			cell_sunday_open = 0
		data_sunday_close.append(cell_sunday_close)
		data_sunday_open.append(cell_sunday_open)

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


		# getting total hours per workday in seconds. it gives us negative float values, but this is fine as it preserves a relationship between rating stars and total hours open.
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


		if cell_tuesday is not None:
			if cell_tuesday_open < cell_tuesday_close:
				cell_tuesday_total_hours = (cell_tuesday_open - cell_tuesday_close)
			elif cell_tuesday_open > cell_tuesday_close:
				cell_tuesday_total_hours = (cell_tuesday_close - cell_tuesday_open)
			else:
				cell_tuesday_total_hours = 0
		else:
			cell_tuesday_total_hours = 0
		data_tuesday_total_hours.append(cell_tuesday_total_hours)

		if cell_wednesday is not None:
			if cell_wednesday_open < cell_wednesday_close:
				cell_wednesday_total_hours = (cell_wednesday_open - cell_wednesday_close)
			elif cell_wednesday_open > cell_wednesday_close:
				cell_wednesday_total_hours = (cell_wednesday_close - cell_wednesday_open)
			else:
				cell_wednesday_total_hours = 0
		else:
			cell_wednesday_total_hours = 0
		data_wednesday_total_hours.append(cell_wednesday_total_hours)

		if cell_thursday is not None:
			if cell_thursday_open < cell_thursday_close:
				cell_thursday_total_hours = (cell_thursday_open - cell_thursday_close)
			elif cell_thursday_open > cell_thursday_close:
				cell_thursday_total_hours = (cell_thursday_close - cell_thursday_open)
			else:
				cell_thursday_total_hours = 0
		else:
			cell_thursday_total_hours = 0
		data_thursday_total_hours.append(cell_thursday_total_hours)

		if cell_friday is not None:
			if cell_friday_open < cell_friday_close:
				cell_friday_total_hours = (cell_friday_open - cell_friday_close)
			elif cell_friday_open > cell_friday_close:
				cell_friday_total_hours = (cell_friday_close - cell_friday_open)
			else:
				cell_friday_total_hours = 0
		else:
			cell_friday_total_hours = 0
		data_friday_total_hours.append(cell_friday_total_hours)

		if cell_saturday is not None:
			if cell_saturday_open < cell_saturday_close:
				cell_saturday_total_hours = (cell_saturday_open - cell_saturday_close)
			elif cell_saturday_open > cell_saturday_close:
				cell_saturday_total_hours = (cell_saturday_close - cell_saturday_open)
			else:
				cell_saturday_total_hours = 0
		else:
			cell_saturday_total_hours = 0
		data_saturday_total_hours.append(cell_saturday_total_hours)


		if cell_sunday is not None:
			if cell_sunday_open < cell_sunday_close:
				cell_sunday_total_hours = (cell_sunday_open - cell_sunday_close)
			elif cell_sunday_open > cell_sunday_close:
				cell_sunday_total_hours = (cell_sunday_close - cell_sunday_open)
			else:
				cell_sunday_total_hours = 0
		else:
			cell_sunday_total_hours = 0
		data_sunday_total_hours.append(cell_sunday_total_hours)


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

# to see how  many days a business is open each week:
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

# the values given are negative, but that's okay because it preserves the relationship between stars and hours



		cell_attribute = json_line.get('attributes')
		# list_attributes.append(1 * cell_attribute)

		if cell_attribute is not None:
			cell_outdoor_seating = cell_attribute.get('OutdoorSeating')
			if cell_outdoor_seating is not None:
				if cell_outdoor_seating == True:
					data_outdoor_seating.append(1)
				else:
					data_outdoor_seating.append(0)
			else:
				data_outdoor_seating.append(0)
		else:
			data_outdoor_seating.append(None)

		# # To get whether the business is by appointment only, 'None' if the business did not specified that of it does not have any attributes.
		if cell_attribute is not None:
			cell_credit_cards = cell_attribute.get('BusinessAcceptsCreditCards')
			if cell_credit_cards is not None:
				if cell_credit_cards == True:
					data_credit_cards.append(1)
				else:
					data_credit_cards.append(0)
			else:
				data_credit_cards.append(0)
		else:
			data_credit_cards.append(None)

		if cell_attribute is not None:
			cell_take_out = cell_attribute.get('RestaurantsTakeOut')
			if cell_take_out is not None:
				if cell_take_out == True:
					data_take_out.append(1)
				else:
					data_take_out.append(0)
			else:
				data_take_out.append(0)
		else:
			data_take_out.append(None)


		if cell_attribute is not None:
			cell_groups = cell_attribute.get('RestaurantsGoodForGroups')
			if cell_groups is not None:
				if cell_groups == True:
					data_groups.append(1)
				else:
					data_groups.append(0)
			else:
				data_groups.append(0)
		else:
			data_groups.append(None)


		# # To get restaurant price range, 'None' if the business did not specified that or it does not have any attributes.
		if cell_attribute is not None:
			cell_restaurant_price = cell_attribute.get('RestaurantsPriceRange2')
			if cell_restaurant_price is not None:
				data_restaurant_price.append(cell_restaurant_price)
			else:
				data_restaurant_price.append(0)
		else:
			data_restaurant_price.append(None)

		if cell_attribute is not None:
			cell_caters = cell_attribute.get('Caters')
			if cell_caters is not None:
				if cell_caters == True:
					data_caters.append(1)
				else:
					data_caters.append(0)
			else:
				data_caters.append(0)
		else:
			data_caters.append(None)

		if cell_attribute is not None:
			cell_reservations = cell_attribute.get('RestaurantsReservations')
			if cell_reservations is not None:
				if cell_reservations == True:
					data_reservations.append(1)
				else:
					data_reservations.append(0)
			else:
				data_reservations.append(0)
		else:
			data_reservations.append(None)

		if cell_attribute is not None:
			cell_kids = cell_attribute.get('GoodForKids')
			if cell_kids is not None:
				if cell_kids == True:
					data_kids.append(1)
				else:
					data_kids.append(0)
			else:
				data_kids.append(0)
		else:
			data_kids.append(None)

		# if cell_attribute is not None:
		# 	cell_tv = cell_attribute.get('HasTV')
		# 	if cell_tv is not None:
		# 		if cell_kids == True:
		# 			data_kids.append(1)
		# 		else:
		# 			data_kids.append(0)
		# 	else:
		# 		data_tv.append(0)
		# else:
		# 	data_tv.append(None)

		if cell_attribute is not None:
			cell_meal = cell_attribute.get('GoodForMeal')
			if cell_meal is not None:
				x = ast.literal_eval(cell_meal)
				if x is not None:
					try:
						cell_dinner = x['dinner']
						if cell_dinner is not None:
							data_dinner.append(1 * cell_dinner)
						else:
							data_dinner.append(0)
					except KeyError:
						data_dinner.append(None)

					try:
						cell_lunch = x['lunch']
						if cell_lunch is not None:
							data_lunch.append(1 * cell_lunch)
						else:
							data_lunch.append(0)
					except KeyError:
						data_lunch.append(None)

					try:
						cell_breakfast = x['breakfast']
						if cell_breakfast is not None:
							data_breakfast.append(1 * cell_breakfast)
						else:
							data_breakfast.append(0)
					except KeyError:
						data_breakfast.append(None)

					try:
						cell_latenight = x['latenight']
						if cell_latenight is not None:
							data_latenight.append(1 * cell_latenight)
						else:
							data_latenight.append(0)
					except KeyError:
						data_latenight.append(None)

					try:
						cell_dessert = x['dessert']
						if cell_dessert is not None:
							data_dessert.append(1 * cell_dessert)
						else:
							data_dessert.append(0)
					except KeyError:
						data_dessert.append(None)

				else:
					data_dinner.append(None)
					data_lunch.append(None)
					data_breakfast.append(None)
					data_latenight.append(None)
					data_dessert.append(None)
			else:
				data_dinner.append(None)
				data_lunch.append(None)
				data_breakfast.append(None)
				data_latenight.append(None)
				data_dessert.append(None)
		else:
			data_dinner.append(None)
			data_lunch.append(None)
			data_breakfast.append(None)
			data_latenight.append(None)
			data_dessert.append(None)


		if cell_attribute is not None:
			cell_alcohol = cell_attribute.get('Alcohol')
			if cell_alcohol is not None:
				if cell_alcohol != "u'none'" and cell_alcohol != "'none'":
					data_alcohol.append(1)
				else:
					data_alcohol.append(0)
			else:
				data_alcohol.append(0)
		else:
			data_alcohol.append(None)

		if cell_attribute is not None:
			cell_noise_level = cell_attribute.get('NoiseLevel')
			if cell_noise_level is not None:
				if cell_noise_level == "u'quiet'" or cell_noise_level == "'quiet'":
					data_noise_quiet.append(1)
				else:
					data_noise_quiet.append(0)
			else:
				data_noise_quiet.append(0)
		else:
			data_noise_quiet.append(None)

		if cell_attribute is not None:
			cell_noise_level = cell_attribute.get('NoiseLevel')
			if cell_noise_level is not None:
				if cell_noise_level == "u'average'" or cell_noise_level == "'average'":
					data_noise_avg.append(1)
				else:
					data_noise_avg.append(0)
			else:
				data_noise_avg.append(0)
		else:
			data_noise_avg.append(None)

		if cell_attribute is not None:
			cell_noise_level = cell_attribute.get('NoiseLevel')
			if cell_noise_level is not None:
				if cell_noise_level == "u'loud'" or cell_noise_level == "'loud'":
					data_noise_loud.append(1)
				else:
					data_noise_loud.append(0)
			else:
				data_noise_loud.append(0)
		else:
			data_noise_loud.append(None)

		if cell_attribute is not None:
			cell_ambience = cell_attribute.get('Ambience')
			if cell_ambience is not None:
				x = ast.literal_eval(cell_ambience)
				if x is not None:
					try:
						cell_touristy = x['touristy']
						if cell_touristy is not None:
							data_touristy.append(1 * cell_touristy)
						else:
							data_touristy.append(0)
					except KeyError:
						data_touristy.append(None)

					try:
						cell_classy = x['classy']
						if cell_classy is not None:
							data_classy.append(1 * cell_classy)
						else:
							data_classy.append(0)
					except KeyError:
						data_classy.append(None)

					try:
						cell_romantic = x['romantic']
						if cell_romantic is not None:
							data_romantic.append(1 * cell_romantic)
						else:
							data_romantic.append(0)
					except KeyError:
						data_romantic.append(None)

					try:
						cell_intimate = x['intimate']
						if cell_intimate is not None:
							data_intimate.append(1 * cell_intimate)
						else:
							data_intimate.append(0)
					except KeyError:
						data_intimate.append(None)

					try:
						cell_hipster = x['hipster']
						if cell_hipster is not None:
							data_hipster.append(1 * cell_hipster)
						else:
							data_hipster.append(0)
					except KeyError:
						data_hipster.append(None)

					try:
						cell_divey = x['divey']
						if cell_divey is not None:
							data_divey.append(1 * cell_divey)
						else:
							data_divey.append(0)
					except KeyError:
						data_divey.append(None)

					try:
						cell_trendy = x['trendy']
						if cell_trendy is not None:
							data_trendy.append(1 * cell_trendy)
						else:
							data_trendy.append(0)
					except KeyError:
						data_trendy.append(None)

					try:
						cell_upscale = x['upscale']
						if cell_upscale is not None:
							data_upscale.append(1 * cell_upscale)
						else:
							data_upscale.append(0)
					except KeyError:
						data_upscale.append(None)

					try:
						cell_casual = x['casual']
						if cell_casual is not None:
							data_casual.append(1 * cell_casual)
						else:
							data_casual.append(0)
					except KeyError:
						data_casual.append(None)

				else:
					data_touristy.append(None)
					data_classy.append(None)
					data_romantic.append(None)
					data_intimate.append(None)
					data_hipster.append(None)
					data_divey.append(None)
					data_trendy.append(None)
					data_upscale.append(None)
					data_casual.append(None)
			else:
				data_touristy.append(None)
				data_classy.append(None)
				data_romantic.append(None)
				data_intimate.append(None)
				data_hipster.append(None)
				data_divey.append(None)
				data_trendy.append(None)
				data_upscale.append(None)
				data_casual.append(None)
		else:
			data_touristy.append(None)
			data_classy.append(None)
			data_romantic.append(None)
			data_intimate.append(None)
			data_hipster.append(None)
			data_divey.append(None)
			data_trendy.append(None)
			data_upscale.append(None)
			data_casual.append(None)


		if cell_attribute is not None:
			cell_parking = cell_attribute.get('BusinessParking')
			if cell_parking is not None:
				x = ast.literal_eval(cell_parking)
				if x is not None:
					try:
						cell_garage = x['garage']
						if cell_garage is not None:
							data_garage.append(1 * cell_garage)
						else:
							data_garage.append(0)
					except KeyError:
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
					data_garage.append(None)
					data_streetparking.append(None)
					data_validated.append(None)
					data_lot.append(None)
					data_valet.append(None)
					data_regular_parking.append(None)
			else:
				data_garage.append(None)
				data_streetparking.append(None)
				data_validated.append(None)
				data_lot.append(None)
				data_valet.append(None)
				data_regular_parking.append(None)
		else:
			data_garage.append(None)
			data_streetparking.append(None)
			data_validated.append(None)
			data_lot.append(None)
			data_valet.append(None)
			data_regular_parking.append(None)

		# cell_categories = json_line.get('categories')
		# list_categories.append(cell_categories)

		# # # To get a variable indicating whether 'restaurant' is one of the items in categories. 
		# if cell_categories is not None:
		# 	cell_restaurant = ('Restaurants' in cell_categories)
		# 	data_restaurant.append(1 * cell_restaurant)
		# else:
		# 	data_restaurant.append(None)

		# # accounting for types of restaurants:
		if cell_categories is not None:
			cell_thai = ('Thai' in cell_categories)
			data_thai.append(1 * cell_thai)
		else:
			data_thai.append(None)

		if cell_categories is not None:
			cell_malaysian = ('Malaysian' in cell_categories)
			data_malaysian.append(1 * cell_malaysian)
		else:
			data_malaysian.append(None)

		if cell_categories is not None:
			cell_irish = ('Irish' in cell_categories)
			data_irish.append(1 * cell_irish)
		else:
			data_irish.append(None)

		if cell_categories is not None:
			cell_steakhouses = ('Steakhouses' in cell_categories)
			data_steakhouses.append(1 * cell_steakhouses)
		else:
			data_steakhouses.append(None)

		if cell_categories is not None:
			cell_burgers = ('Burgers' in cell_categories)
			data_burgers.append(1 * cell_burgers)
		else:
			data_burgers.append(None)

		if cell_categories is not None:
			cell_fast_food = ('Fast Food' in cell_categories)
			data_fast_food.append(1 * cell_fast_food)
		else:
			data_fast_food.append(None)

		if cell_categories is not None:
			cell_french = ('French' in cell_categories)
			data_french.append(1 * cell_french)
		else:
			data_french.append(None)

		if cell_categories is not None:
			cell_modern_european = ('Modern European' in cell_categories)
			data_modern_european.append(1 * cell_modern_european)
		else:
			data_modern_european.append(None)		

		if cell_categories is not None:
			cell_seafood = ('Seafood' in cell_categories)
			data_seafood.append(1 * cell_seafood)
		else:
			data_seafood.append(None)

		if cell_categories is not None:
			cell_mexican = ('Mexican' in cell_categories)
			data_mexican.append(1 * cell_mexican)
		else:
			data_mexican.append(None)

		if cell_categories is not None:
			cell_bars = ('Bars' in cell_categories)
			data_bars.append(1 * cell_bars)
		else:
			data_bars.append(None)

		if cell_categories is not None:
			cell_gastropubs = ('Gastropubs' in cell_categories)
			data_gastropubs.append(1 * cell_gastropubs)
		else:
			data_gastropubs.append(None)

		if cell_categories is not None:
			cell_chinese = ('Chinese' in cell_categories)
			data_chinese.append(1 * cell_chinese)
		else:
			data_chinese.append(None)

		if cell_categories is not None:
			cell_data_american_new = ('American (New)' in cell_categories)
			data_american_new.append(1 * cell_data_american_new)
		else:
			data_american_new.append(None)

		if cell_categories is not None:
			cell_american_traditional = ('American (Traditional)' in cell_categories)
			data_american_traditional.append(1 * cell_american_traditional)
		else:
			data_american_traditional.append(None)

		if cell_categories is not None:
			cell_japanese = ('Japanese' in cell_categories)
			data_japanese.append(1 * cell_japanese)
		else:
			data_japanese.append(None)

		if cell_categories is not None:
			cell_pizza = ('Pizza' in cell_categories)
			data_pizza.append(1 * cell_pizza)
		else:
			data_pizza.append(None)

		if cell_categories is not None:
			cell_mediterranean = ('Mediterranean' in cell_categories)
			data_mediterranean.append(1 * cell_mediterranean)
		else:
			data_mediterranean.append(None)

		if cell_categories is not None:
			cell_breakfast_brunch = ('Breakfast & Brunch' in cell_categories)
			data_breakfast_brunch.append(1 * cell_breakfast_brunch)
		else:
			data_breakfast_brunch.append(None)

		if cell_categories is not None:
			cell_delis = ('Delis' in cell_categories)
			data_delis.append(1 * cell_delis)
		else:
			data_delis.append(None)

		if cell_categories is not None:
			cell_italian = ('Italian' in cell_categories)
			data_italian.append(1 * cell_italian)
		else:
			data_italian.append(None)

		if cell_categories is not None:
			cell_diners = ('Diners' in cell_categories)
			data_diners.append(1 * cell_diners)
		else:
			data_diners.append(None)

		if cell_categories is not None:
			cell_vegan = ('Vegan' in cell_categories)
			data_vegan.append(1 * cell_vegan)
		else:
			data_vegan.append(None)

		if cell_categories is not None:
			cell_latin_american = ('Latin American' in cell_categories)
			data_latin_american.append(1 * cell_latin_american)
		else:
			data_latin_american.append(None)

		if cell_categories is not None:
			cell_german = ('German' in cell_categories)
			data_german.append(1 * cell_german)
		else:
			data_german.append(None)

		if cell_categories is not None:
			cell_coffee_tea = ('Coffee & Tea' in cell_categories)
			data_coffee_tea.append(1 * cell_coffee_tea)
		else:
			data_coffee_tea.append(None)

		if cell_categories is not None:
			cell_tapas = ('Tapas\/Small Plates' in cell_categories)
			data_tapas.append(1 * cell_tapas)
		else:
			data_tapas.append(None)

		if cell_categories is not None:
			cell_bakeries = ('Bakeries' in cell_categories)
			data_bakeries.append(1 * cell_bakeries)
		else:
			data_bakeries.append(None)


dataset = pandas.DataFrame(data={'name': data_name,
					'text_analysis_prediction': data_prediction,
					# 'address': data_address,
					# 'state': data_state,
					# 'city': data_city,
					'Las_Vegas': data_vegas, 
					'Toronto': data_toronto, 
					'Phoenix': data_phoenix, 
					'Charlotte': data_clt, 
					'Scottsdale': data_scottsdale, 
					'Calgary': data_calgary, 
					'Pittsburgh': data_pitt, 
					'Montréal': data_montreal, 
					'Mesa': data_mesa, 
					'Henderson': data_henderson, 
					'Tempe': data_tempe,
					'Chandler': data_chandler, 
					'Cleveland': data_cleveland, 
					'Glendale': data_glendale, 
					'Madison': data_madison,
					'AB': data_AB,
					'AZ': data_AZ,
					'IL': data_IL,
					'NC': data_NC,
					'NV': data_NV,
					'NY': data_NY,
					'OH': data_OH,
					'ON': data_ON,
					'PA': data_PA,
					'QC': data_QC,
					'SC': data_SC,
					'WI': data_WI,
					'CAN': data_CAN,
					'UK': data_UK,
					'US': data_US,
					'latitude': data_latitude,
					'longitude': data_longitude,
					# 'hours': data_hours,
					# 'mon': data_monday,
					# 'tues': data_tuesday,
					# 'weds': data_wednesday,
					# 'thurs': data_thursday,
					# 'fri': data_friday,
					# 'sat': data_saturday,
					# 'sun': data_sunday,
					'open_on_weekends': data_weekends,
					'mon_close': data_monday_close,
					'tues_close': data_tuesday_close,
					'weds_close': data_wednesday_close,
					'thurs_close': data_thursday_close,
					'fri_close': data_friday_close,
					'sat_close': data_saturday_close,
					'sun_close': data_sunday_close,
					'mon_open': data_monday_open,
					'tues_open': data_tuesday_open,
					'weds_open': data_wednesday_open,
					'thurs_open': data_thursday_open,
					'fri_open': data_friday_open,
					'sat_open': data_saturday_open,
					'sun_open': data_sunday_open,
					'monday_total_hours': data_monday_total_hours,
					'tuesday_total_hours': data_tuesday_total_hours,
					'wednesday_total_hours': data_wednesday_total_hours,
					'thursday_total_hours': data_thursday_total_hours,
					'friday_total_hours': data_friday_total_hours,
					'saturday_total_hours': data_saturday_total_hours,
					'sunday_total_hours': data_sunday_total_hours,
					'days_open': data_days_open,
					'mode_open_time': data_open_mode,
					'mode_close_time': data_close_mode,
					'mean_hours_per_day': data_mean_hours,
					'hours_per_week': data_total_hours,
					# 'attributes_list': list_attributes, 13^
					'outdoor_seating': data_outdoor_seating,
					'credit_cards': data_credit_cards,
					'take_out': data_take_out,
					'groups': data_groups,
					'restaurant_price': data_restaurant_price,
					# # couldn't get rest price to load as an integer.
					'caters': data_caters,
					'reservations': data_reservations,
					'kids': data_kids,
					# 'has_tv': data_tv,
					# didn't include has tv
					'dinner': data_dinner,
					'lunch': data_lunch,
					'breakfast': data_breakfast,
					'latenight': data_latenight,
					'dessert': data_dessert,
					'has_alcohol': data_alcohol,
					'noise_quiet': data_noise_quiet,
					'noise_avg': data_noise_avg,
					'noise_loud': data_noise_loud,
					'touristy': data_touristy,
					'classy': data_classy,
					'romantic': data_romantic,
					'intimate': data_intimate,
					'hipster': data_hipster,
					'divey': data_divey,
					'trendy': data_trendy,
					'upscale': data_upscale,
					'casual': data_casual,
					'garage': data_garage,
					'street_parking': data_streetparking,
					'validated': data_validated,
					'lot': data_lot,
					'valet': data_valet,
					'regular_parking': data_regular_parking,
					'is_restaurant': data_restaurant,
					# 'categories_list': list_categories,
					'thai': data_thai,
					'malaysian': data_malaysian,
					'irish': data_irish,
					'steakhouses': data_steakhouses,
					'burgers': data_burgers,
					'fast_food': data_fast_food,
					'data_french': data_french,
					'modern_european': data_modern_european,
					'seafood': data_seafood,
					'mexican': data_mexican,
					'bars': data_bars,
					'gastropubs': data_gastropubs,
					'chinese': data_chinese,
					'american_new': data_american_new,
					'american_traditional': data_american_traditional,
					'japanese': data_japanese,
					'pizza': data_pizza,
					'mediterranean': data_mediterranean,
					'breakfast_brunch': data_breakfast_brunch,
					'delis': data_delis,
					'italian': data_italian,
					'diners': data_diners,
					'vegan': data_vegan,
					'latin_america': data_latin_american,
					'german': data_german,
					'coffee_tea': data_coffee_tea,
					'tapas': data_tapas,
					'bakeries': data_bakeries
						})

data_machine_prediction = []
# creating an empty structure to hold the machine's predictions

dataset = dataset[(dataset['is_restaurant']==1)]
# making sure our data is for restaurant businesses.
# dropping missing values:
dataset = dataset.dropna()
dataset = dataset.replace(to_replace='None',value=np.nan).dropna()


# for row in dataset['name']
data_name = dataset['name']
data = dataset.iloc[:,1:].values	
file = open("restaurants_serialized_random_forest_text.pickle", "rb")
# this opens the random forest that I trained earlier
machine = pickle.load(file)
# this accesses the RF machine I trained earlier.		
cell_machine_prediction = machine.predict(data)	

df1 = pandas.DataFrame(data={'name': data_name,
	'predicted_stars': cell_machine_prediction})

print(df1)

df1.to_csv('restaurants_predictions_with_text.csv')




