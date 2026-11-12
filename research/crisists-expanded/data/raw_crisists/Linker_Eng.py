import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import argparse



# this function put on the good format the date from the nlp dataset
def to_date(my_date):
	set_month= {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
				'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
	month = set_month[my_date[4:7]]
	day = my_date[8:10]
	year = my_date[26:30]
	final_date = year+'-'+month+'-'+day
	return final_date


# open the knowledge base of the corresponding state with the corresponding category
def open_state_file(dataset_path: str, STATE: str, CATEGORY: str) -> list[str]:
	tags = set()
	with open('./utils/Keywords/'+STATE+'_keywords_'+CATEGORY+'_no_dupe.txt', 'r') as fp:
		for line in fp:
			# remove linebreak from a current name
			# linebreak is the last character of each line
			x = line[:-1]
			# add current item to the list
			tags.add(x.lower())
	return tags


# the function take a time series dataframe (df) and return another dataset with all the data put in one parameter
# (Window) except the date and the label which are in different columns
# window size is equal to the size of the window

def window_creation(df: object, window_size: int, timestamp_colum: str) -> object:
	"""
	Create a TS window for each date. Distance of sliding window is 1.

	df: Dataframe of a whole TS record in one state. Each row represents the record on one date.
	window_size: Size of TS window.
	label_column: Name of the column of label
	timestamp_colum: Name of the column of time stamp
	"""
	df = df.reset_index()
	df = df.drop(columns=['index'])
	good_columns = list(df.columns)
	good_columns.remove(timestamp_colum)
	window_df = df[good_columns]
	scaler = StandardScaler()
	window_df = pd.DataFrame(scaler.fit_transform(window_df), columns=window_df.columns)
	date_df = df[timestamp_colum]
	window_data = []
	label_data = []
	date_data = []
	for i in range(len(df)-window_size):
		my_window = window_df.loc[i:i+window_size-1]
		window_data.append(my_window.to_numpy())
		date_data.append((date_df.loc[i+window_size-1]))
	return_value = pd.DataFrame({'Date': date_data, 'Window': window_data})
	return return_value


# the function that link the NLP data and the time series data for English dataset
# crisis_to_link is one row from crisis knowledge
# directory is the path of NLP data
# dataset_path  is the path of time series data
# window_size is the size of the window for time series data
# fillNA is the value to fill the missing value
def linker(
		crisis_to_link: object,
		corpus_path: str,  # path of corpus directory
		ts_dataset_path: str,  # path of ts data directory
		window_size: int,
		date_column: str,
		fillNA : int) -> object:

	# list of states associated with the crisis
	type_of_crisis = crisis_to_link['Crisis']


	states = crisis_to_link['Places'].replace('[', '').replace(']', '').split(',')

	# features to keep
	features = ['date', 'mean_speed_wind', 'precipitation', 'snow_fall', 'snow_depth', 'temp_max', 'temp_min', 'temp_mean',

				'wind_dir_angle_2min', 'wind_dir_angle_5sec', 'wind_max_speed_2min', 'wind_max_speed_5sec']

	df_return = pd.DataFrame({'Date': [], 'Text': [],'Crisis_Type' :[], 'Window': [],  'label_humanitarian': [], 'label_useful' : [], 'label_urgent' : [], 'label_sudden' : []})

	for state in states:
		# Open the time series file corresponding to the state
		ts_df = pd.read_csv(ts_dataset_path + '/' + state + '_Time_Series_Data.csv')
		ts_df = ts_df[features].sort_values(by='date')  # Frequency of TS is per day.
		ts_df = ts_df.fillna(fillNA)
		# Window Creation
		window_df = window_creation(ts_df, window_size, date_column)
		#since New Zealand has no keyword, we skip this part
		if state != 'New_Zealand' :
			#  we get the keywords to link the location mention to the state
			tags_city = open_state_file(ts_dataset_path, state, 'city')
			tags_county = open_state_file(ts_dataset_path, state, 'county')
			tags_state = open_state_file(ts_dataset_path, state, 'state')
		date = []
		text = []
		label_humanitarian = []
		label_useful = []
		label_urgent = []
		label_sudden = []
		window_data = []
		crisis_label = []

		# Process NLP data and link to time series data
		for root, dirs, files in os.walk(corpus_path + '/' + crisis_to_link['Path_name']):
			for fil in files:
				if fil.endswith('.jsonl'):
					with open(os.path.join(root,fil), 'r') as json_file:
						json_list = list(json_file)  # a list of '{}', each {} is a tweet and its features.
						for json_str in json_list:  # for each tweet loaded
							result = json.loads(json_str)
							place = result['location_mentions']

					# if the location mention is empty (the tweet does not refer to particular place),
					# thanks to the crisis_to_link, we know which crisis this tweet make reference
					# (the tweet speak about this crisis) so we assume that if location mention is empty
					# we assume that the tweet make a reference to the current state since this state is the localisation of the crisis
							#we still have the New Zealand special case
							if place == [] or state == 'New_Zealand':
								# Put NLP date on the same format as time series date
								date_NLP = to_date(result['created_at'])
								# Check if there is matching date between time series and tweets.
								if list(window_df['Window'][ window_df['Date'] == date_NLP]) != [] :
									date.append(date_NLP)
									text.append(result['text'])
									linked_data = window_df[window_df['Date'] == date[-1]]
									my_window = list(linked_data['Window'])[0]
									window_data.append(my_window)
									# for the label, we take reference from the time series label
									label_humanitarian.append(result['humAID_class'])
									crisis_label.append(type_of_crisis)
									if result['humAID_class'] == 'not_humanitarian' :
										label_useful.append('not_humanitarian')
										label_urgent.append('not_humanitarian')
									else :
										label_useful.append('useful')
										if result['humAID_class'] == 'rescue_volunteering_or_donation_effort' :
											label_urgent.append('not_urgent')
										else :
											label_urgent.append('urgent')

									if result['humAID_class'] == 'not_humanitarian' :
										label_sudden.append('Not_Crisis_period')
									else :
										if type_of_crisis == 'Earthquake' or type_of_crisis == 'WildFire' :
											label_sudden.append('Sudden_Crisis')
										else :
											label_sudden.append('Non_Sudden_Crisis')

							else:
								for zone in place:
									# if the location mention refer to a state and this reference is in our knowledge database
									bool1 = zone['type'] == 'State' and zone['text'].lower() in tags_state
									# if the location mention refer to a county and this reference is in our knowledge database
									bool2 = zone['type'] == 'County' and zone['text'].lower() in tags_county
									# if the location mention refer to a city and this reference is in our knowledge database
									bool3 = zone['type'] == 'City/town' and zone['text'].lower() in tags_city
									if bool1 or bool2 or bool3:
										date.append(to_date(result['created_at']))
										text.append(result['text'])
										linked_data = window_df[window_df['Date'] == date[-1]]
										my_window = list(linked_data['Window'])[0]
										window_data.append(my_window)
										label_humanitarian.append(result['humAID_class'])
										crisis_label.append(type_of_crisis)
										if result['humAID_class'] == 'not_humanitarian' :
											label_useful.append('not_humanitarian')
											label_urgent.append('not_humanitarian')
										else :
											label_useful.append('useful')
											if result['humAID_class'] == 'rescue_volunteering_or_donation_effort' :
												label_urgent.append('not_urgent')
											else :
												label_urgent.append('urgent')

										if result['humAID_class'] == 'not_humanitarian' :
											label_sudden.append('Not_Crisis_period')
										else :
											if type_of_crisis == 'Earthquake' or type_of_crisis == 'WildFire' :
												label_sudden.append('Sudden_Crisis')
											else :
												label_sudden.append('Non_Sudden_Crisis')

		df = pd.DataFrame({'Date': date, 'Text': text,'Crisis_Type' :crisis_label, 'Window': window_data, 'label_humanitarian': label_humanitarian, 'label_useful' : label_useful, 'label_urgent' : label_urgent, 'label_sudden' : label_sudden})
		df_return = pd.concat([df_return, df])
	return df_return

# loading file
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window_size",type = int, default = 5, help="size (in number of days) of our daily time series data")
parser.add_argument("-o", "--output_file",type = str, default = './Multi_modal_dataset/CrisisTS_Eng_Personnalized.csv', help="name of your output file")
args = parser.parse_args()

directory_nlp = './Textual_Data/English_Corpus'
directory_time_series = './Time_Series/EnglishTS'
path_knowledge = './utils/crisis_knowledge_eng.csv'
knowledge = pd.read_csv(path_knowledge, sep='\t')  # (9, 3), 3 columns represent Crisis, Places, Path_name (tweets)

# for all test crisis in english
All_possible_crisis = ['Hurricane', 'WildFire', 'Flood','Earthquake']
Test_crisis = 'Hurricane'

Train_knowledge = knowledge[knowledge['Crisis'] != Test_crisis]
Test_knowledge = knowledge[knowledge['Crisis'] == Test_crisis]

# link
multi_modal_test_data = pd.DataFrame({'Date': [], 'Text': [],'Crisis_Type' :[], 'Window': [],  'label_humanitarian': [], 'label_useful' : [], 'label_urgent' : [], 'label_sudden' : []})
for index, crisis in knowledge.iterrows():
	print('Linking textual data and meteorological data of '+crisis['Path_name']+ ' ...')
	multi_modal_test_data = pd.concat([multi_modal_test_data,
									   linker(crisis, directory_nlp,
											  directory_time_series,
											  args.window_size,
											  'date',
											  0)])


print('Done !')
multi_modal_test_data.to_csv(args.output_file,sep = '\t')
