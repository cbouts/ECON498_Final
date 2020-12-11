import pandas
import json
import ast
from sklearn.impute import SimpleImputer
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
# location:
data_name = []
data_stars = []
data_prediction = []
data_address = []
data_city = []
# the states we look at are those represented by at least 900 businesses. States not appearing on these lists are those which are represented by only 1-19 businesses each. this is likely not enough data to meaningfully contribute to prediction, and it would lead to very poor external validity for these variables.
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

# accounting for business hours:
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

# attributes: 
list_attributes = []
data_credit_cards = []
data_appointment_only = []
data_accepts_insurance = []
data_kids = []

# information about parking: (only include parking_regular and valet in data)
data_garage = []
data_streetparking = []
data_validated = []
data_lot = []
data_valet = []
data_regular_parking = []

# categories
list_categories = []
data_restaurant = []
data_medical = []
data_beauty = []
data_shopping = []
data_local_services = []
data_travel = []
data_arts_entertainment = []
data_home_services = []
data_professional_services = []
data_automotive = []
data_education = []
data_fitness_and_instruction = []
data_events = []
data_nightlife = []




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

		# # text analysis
		cell_categories = json_line.get('categories')
		list_categories.append(cell_categories)

		if cell_categories is not None:
			cell_restaurant = (1 * ('Restaurants' in cell_categories))
			data_restaurant.append(cell_restaurant)
		else:
			data_restaurant.append(None)
		# text analysis

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
			# this appends an empty value for data_prediction for restaurants. 


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
		data_state.append(cell_state)
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
		list_attributes.append(cell_attribute)

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

		if cell_attribute is not None:
			cell_appointment_only = cell_attribute.get('ByAppointmentOnly')
			if cell_appointment_only is not None:
				if cell_appointment_only == True:
					data_appointment_only.append(1)
				else:
					data_appointment_only.append(0)
			else:
				data_appointment_only.append(0)
		else:
			data_appointment_only.append(None)

		if cell_attribute is not None:
			cell_accepts_insurance = cell_attribute.get('AcceptsInsurance')
			if cell_accepts_insurance is not None:
				if cell_accepts_insurance == True:
					data_accepts_insurance.append(1)
				else:
					data_accepts_insurance.append(0)
			else:
				data_accepts_insurance.append(0)
		else:
			data_accepts_insurance.append(None)

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


		# cell_categories = json_line.get('categories')
		# list_categories.append(cell_categories)

		# if cell_categories is not None:
		# 	cell_restaurant = (1 * ('Restaurants' in cell_categories))
		# 	data_restaurant.append(cell_restaurant)
		# else:
		# 	data_restaurant.append(None)

		if cell_categories is not None:
			cell_medical = (1 * ('Health & Medical' in cell_categories))
			data_medical.append(cell_medical)
		else:
			data_medical.append(None)

		if cell_categories is not None:
			cell_beauty = (1 * ('Beauty & Spas' in cell_categories))
			data_beauty.append(cell_beauty)
		else:
			data_beauty.append(None)

		if cell_categories is not None:
			cell_shopping = (1 * ('Shopping' in cell_categories))
			data_shopping.append(cell_shopping)
		else:
			data_shopping.append(None)

		if cell_categories is not None:
			cell_local_services = (1 * ('Local Services' in cell_categories))
			data_local_services.append(cell_local_services)
		else:
			data_local_services.append(None)

		if cell_categories is not None:
			cell_travel = (1 * ('Hotels & Travel' in cell_categories))
			data_travel.append(cell_travel)
		else:
			data_travel.append(None)

		if cell_categories is not None:
			cell_arts_entertainment = (1 * ('Arts & Entertainment' in cell_categories))
			data_arts_entertainment.append(cell_arts_entertainment)
		else:
			data_arts_entertainment.append(None)

		if cell_categories is not None:
			cell_home_services = (1 * ('Home Services' in cell_categories))
			data_home_services.append(cell_home_services)
		else:
			data_home_services.append(None)


		# # To get a variable indicating whether 'professional services' is one of the items in categories. 
		if cell_categories is not None:
			cell_professional_services = (1 * ('Professional Services' in cell_categories))
			data_professional_services.append(cell_professional_services)
		else:
			data_professional_services.append(None)


		# # To get a variable indicating whether 'automotive' is one of the items in categories. 
		if cell_categories is not None:
			cell_automotive = (1 * ('Automotive' in cell_categories))
			data_automotive.append(cell_automotive)
		else:
			data_automotive.append(None)


		# # To get a variable indicating whether 'education' is one of the items in categories. 
		if cell_categories is not None:
			cell_education = (1 * ('Education' in cell_categories))
			data_education.append(cell_education)
		else:
			data_education.append(None)


		# # To get a variable indicating whether 'fitness and instruction' is one of the items in categories. 
		if cell_categories is not None:
			cell_fitness_and_instruction = (1 * ('Fitness & Instruction' in cell_categories))
			data_fitness_and_instruction.append(cell_fitness_and_instruction)
		else:
			data_fitness_and_instruction.append(None)


		# # To get a variable indicating whether 'event planning and services' is one of the items in categories. 
		if cell_categories is not None:
			cell_events= (1 * ('Event Planning & Services' in cell_categories))
			data_events.append(cell_events)
		else:
			data_events.append(None)


		# # To get a variable indicating whether 'event planning and services' is one of the items in categories. 
		if cell_categories is not None:
			cell_nightlife= (1 * ('Nightlife' in cell_categories))
			data_nightlife.append(cell_nightlife)
		else:
			data_nightlife.append(None)


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
					# 'UK': data_UK,
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
					# # 'attributes_list': list_attributes,
					'credit_cards': data_credit_cards,
					'appointment_only': data_appointment_only,
					'accepts_insurance': data_accepts_insurance,
					'good_for_kids': data_kids,
					'garage': data_garage,
					'street_parking': data_streetparking,
					'validated': data_validated,
					'lot': data_lot,
					'valet':data_valet,
					'regular_parking': data_regular_parking,
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


dataset = dataset[(dataset['is_restaurant']!=1)]
# dropping missing values:
dataset = dataset.dropna()
dataset = dataset.replace(to_replace='None',value=np.nan).dropna()


print("dataframe created")
# print(dataset.dtypes)
# note: this gives us data for 110,588 businesses. 110,179 of these have categories that do not include "restaurants," while the other 409 of these have nothing listed for "cell_categories". runme_restaurants.py gives us data for the 49,412 businesses that ARE categorized as restaurants.
# the non_restaurant.py therefore makes predictions for businesses that have categories that do not include "restaurants" AND businesses which may be restaurants or other types of business but are unknown because nothing is listed in "cell_categories".

# train the machine using 
dataset = dataset[(dataset['stars']==1)|(dataset['stars']==2)|(dataset['stars']==3)|(dataset['stars']==4)|(dataset['stars']==5)]

# training our random forest model:
target = dataset.iloc[:,1].values
# this sets stars as the target.
data = dataset.iloc[:,2:].values

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size=0.25)
machine = RandomForestClassifier(n_estimators=800, criterion="gini", max_depth=17)
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

machine.fit(data_training, target_training)

prediction = machine.predict(data_test)
print("Confusion matrix from nonrestaurant random forest:")
print(confusion_matrix(target_test, prediction))

# # print(machine)
print("Accuracy score from nonrestaurant random forest:")
print(accuracy_score(target_test, prediction))

with open("nonrestaurants_serialized_random_forest_text.pickle", "wb") as file:
	pickle.dump(machine, file)
	# dump() serializes to an open file (file-like object), dumps() serializes to a string.


# linear for comparison:

machine = linear_model.LinearRegression()
machine.fit(data_training, target_training)
with open("nonrestaurants_serialized_linear_text.pickle", "wb") as file:
	# formerly nonrestaurants_serialized_lineart.pickle
	pickle.dump(machine, file)

prediction = machine.predict(data_test)
print("..............")
print("r2 score from nonrestaurant linear model:")  
print(r2_score(target_test, prediction))



# ___________________________________________________________________________
# PART B: PREDICTING STARS USING OUR TRAINED MODEL.

# to do this, we copy and paste the file-reading/dataframe construction sections of our earlier code, remove infomration about "stars" from the code (since the .json file doesn't include stars), and change the name of the file being read to the file containing businesses we wish to predict stars for.

# location:
data_name = []
data_prediction = []
data_address = []
data_city = []
# the states we look at are those represented by at least 900 businesses. States not appearing on these lists are those which are represented by only 1-19 businesses each. this is likely not enough data to meaningfully contribute to prediction, and it would lead to very poor external validity for these variables.
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

# accounting for business hours:
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

# attributes: 
list_attributes = []
data_credit_cards = []
data_appointment_only = []
data_accepts_insurance = []
data_kids = []

# information about parking: (only include parking_regular and valet in data)
data_garage = []
data_streetparking = []
data_validated = []
data_lot = []
data_valet = []
data_regular_parking = []

# categories
list_categories = []
data_restaurant = []
data_medical = []
data_beauty = []
data_shopping = []
data_local_services = []
data_travel = []
data_arts_entertainment = []
data_home_services = []
data_professional_services = []
data_automotive = []
data_education = []
data_fitness_and_instruction = []
data_events = []
data_nightlife = []




with open('business_no_stars_review.json', encoding="utf8") as f:
	for line in f:
		json_line = json.loads(line)

		cell_name = json_line.get('name')
		data_name.append(str(cell_name))

		# # WE DON'T GET STARS HERE BECAUSE WE'RE USING A .JSON FILE WITH NO STARS INFORMATION TO PREDICT STARS

		cell_categories = json_line.get('categories')
		list_categories.append(cell_categories)

		if cell_categories is not None:
			cell_restaurant = (1 * ('Restaurants' in cell_categories))
			data_restaurant.append(cell_restaurant)
		else:
			data_restaurant.append(None)
		# text analysis

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
			# this appends an empty value for data_prediction for restaurants. 


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
		data_state.append(cell_state)
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




		cell_attribute = json_line.get('attributes')
		list_attributes.append(cell_attribute)

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

		if cell_attribute is not None:
			cell_appointment_only = cell_attribute.get('ByAppointmentOnly')
			if cell_appointment_only is not None:
				if cell_appointment_only == True:
					data_appointment_only.append(1)
				else:
					data_appointment_only.append(0)
			else:
				data_appointment_only.append(0)
		else:
			data_appointment_only.append(None)

		if cell_attribute is not None:
			cell_accepts_insurance = cell_attribute.get('AcceptsInsurance')
			if cell_accepts_insurance is not None:
				if cell_accepts_insurance == True:
					data_accepts_insurance.append(1)
				else:
					data_accepts_insurance.append(0)
			else:
				data_accepts_insurance.append(0)
		else:
			data_accepts_insurance.append(None)

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



		if cell_categories is not None:
			cell_medical = (1 * ('Health & Medical' in cell_categories))
			data_medical.append(cell_medical)
		else:
			data_medical.append(None)

		if cell_categories is not None:
			cell_beauty = (1 * ('Beauty & Spas' in cell_categories))
			data_beauty.append(cell_beauty)
		else:
			data_beauty.append(None)

		if cell_categories is not None:
			cell_shopping = (1 * ('Shopping' in cell_categories))
			data_shopping.append(cell_shopping)
		else:
			data_shopping.append(None)

		if cell_categories is not None:
			cell_local_services = (1 * ('Local Services' in cell_categories))
			data_local_services.append(cell_local_services)
		else:
			data_local_services.append(None)

		if cell_categories is not None:
			cell_travel = (1 * ('Hotels & Travel' in cell_categories))
			data_travel.append(cell_travel)
		else:
			data_travel.append(None)

		if cell_categories is not None:
			cell_arts_entertainment = (1 * ('Arts & Entertainment' in cell_categories))
			data_arts_entertainment.append(cell_arts_entertainment)
		else:
			data_arts_entertainment.append(None)

		if cell_categories is not None:
			cell_home_services = (1 * ('Home Services' in cell_categories))
			data_home_services.append(cell_home_services)
		else:
			data_home_services.append(None)


		# # To get a variable indicating whether 'professional services' is one of the items in categories. 
		if cell_categories is not None:
			cell_professional_services = (1 * ('Professional Services' in cell_categories))
			data_professional_services.append(cell_professional_services)
		else:
			data_professional_services.append(None)


		# # To get a variable indicating whether 'automotive' is one of the items in categories. 
		if cell_categories is not None:
			cell_automotive = (1 * ('Automotive' in cell_categories))
			data_automotive.append(cell_automotive)
		else:
			data_automotive.append(None)


		# # To get a variable indicating whether 'education' is one of the items in categories. 
		if cell_categories is not None:
			cell_education = (1 * ('Education' in cell_categories))
			data_education.append(cell_education)
		else:
			data_education.append(None)


		# # To get a variable indicating whether 'fitness and instruction' is one of the items in categories. 
		if cell_categories is not None:
			cell_fitness_and_instruction = (1 * ('Fitness & Instruction' in cell_categories))
			data_fitness_and_instruction.append(cell_fitness_and_instruction)
		else:
			data_fitness_and_instruction.append(None)


		# # To get a variable indicating whether 'event planning and services' is one of the items in categories. 
		if cell_categories is not None:
			cell_events= (1 * ('Event Planning & Services' in cell_categories))
			data_events.append(cell_events)
		else:
			data_events.append(None)


		# # To get a variable indicating whether 'event planning and services' is one of the items in categories. 
		if cell_categories is not None:
			cell_nightlife= (1 * ('Nightlife' in cell_categories))
			data_nightlife.append(cell_nightlife)
		else:
			data_nightlife.append(None)


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
					# 'UK': data_UK,
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
					# # 'attributes_list': list_attributes,
					'credit_cards': data_credit_cards,
					'appointment_only': data_appointment_only,
					'accepts_insurance': data_accepts_insurance,
					'good_for_kids': data_kids,
					'garage': data_garage,
					'street_parking': data_streetparking,
					'validated': data_validated,
					'lot': data_lot,
					'valet':data_valet,
					'regular_parking': data_regular_parking,
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

data_machine_prediction = []
# creating an empty structure to hold the machine's predictions

dataset = dataset[(dataset['is_restaurant']!=1)]
# making sure our data is for non-restaurant businesses.
# dropping missing values:
dataset = dataset.dropna()
dataset = dataset.replace(to_replace='None',value=np.nan).dropna()


data_name = dataset['name']
data = dataset.iloc[:,1:].values	
file = open("nonrestaurants_serialized_random_forest_text.pickle", "rb")
# this opens the random forest that I trained earlier
machine = pickle.load(file)
# this accesses the RF machine I trained earlier.		
cell_machine_prediction = machine.predict(data)	

df1 = pandas.DataFrame(data={'name': data_name,
	'predicted_stars': cell_machine_prediction})

print(df1)

df1.to_csv('nonrestaurants_predictions_with_text.csv')




