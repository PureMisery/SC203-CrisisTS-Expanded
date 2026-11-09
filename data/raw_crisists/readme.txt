Documentation for CrisisTS:

1.Time_Series: 
	It contains two repositories: EnglishTS and FrenchTS
	1.1 EnglishTS: 
		It contains all the time series we used for alignment with IDRISI-RE. 
		The files inside EnglishTS have the following name format: [State]_Time_Series_Data.csv
		All the CSV files inside EnglishTS, except New_Zealand_Time_Series_Data.csv, contains the data in the following format: [date, mean_speed_wind, evaporation, min_temp_water, max_temp_water, precipitation, sun_percent, min_temp_ground, max_temp_ground, snow_fall, snow_depth, temp_max, temp_min, temp_mean, wind_dir_angle_2min, wind_dir_angle_5sec, wind_max_speed_2min, wind_max_speed_5sec, is_fog, is_big_fog, is_thunder, is_snow, is_hail, is_glaze, is_dust, is_haze, is_blowing_snow, is_tornado]
		New_Zealand_Time_Series_Data.csv contains the data in the following format: [date, precipitation, wind_dir_angle_2min, mean_speed_wind, mean_speed_wind.1, temp_mean, temp_max, temp_min,	 wind_dir_angle_5sec, snow_fall, snow_depth, wind_max_speed_5sec, wind_max_speed_2min]

	1.2 FrenchTS: 
		It contains a sample of six months of the time series we used for alignment with Kozlowski. 
		The files inside FrenchTS have the following name format: [yyyy]/synop.[yyyymm].csv
		The csv file contains the data on the following format: [numer_sta, date, pmer, tend, cod_tend, dd, ff, t, td, u, vv, ww, w1, w2, n, nbas, hbas, cl, cm, ch, pres, niv_bar, geop, tend24, tn12, tn24, tx12, tx24, tminsol, sw, tw, raf10, rafper, per, etat_sol, ht_neige, ssfrai, perssfrai, rr1, rr3, rr6, rr12, rr24, phenspe1, phenspe2, phenspe3, phenspe4, nnuage1, ctype1, hnuage1, nnuage2, ctype2, hnuage2, nnuage3, ctype3, hnuage3, nnuage4, ctype4, hnuage4]

	
2.Textual_Data: 
	Contains two repositories: IDRISI-RE and Kozlowski
	2.1 IDRISI-RE: 
		It contains all the English textual data from IDRISI-RE. 
		The files inside IDRISI-RE have the following name format: [crisis_name]_[crisis_year]/dev.jsonl or [crisis_name]_[crisis_year]/train.jsonl. 
		The json contains the data in the following format: {"tweet_id": int, "user_id": int, "text": str, "created_at": date, "humAID_class": str, "location_mentions": list(str)}
	
	2.2 Kozlowski: 
		It contains all the French textual data from Kozlowski in a unique CSV file. 
		The CSV file contains the data in the following format: [crisis_type, type_crisis, event, utility, urgency, humanitarian, text, created_at, date,time]
	
3.Multi_modal_dataset: 
	It contains two repositories: Multi_modal_ENG and Multi_modal_FR
	3.1 Multi_modal_ENG:
		It contains all the english multimodal data we used for our experiments in a csv file.
		The CSV file contains the data in the following format: [date, text, time_series, crisis_type, utility, urgency, humanitarian, sudden]
	3.2 Multi_modal_FR:
		It contains all the french multimodal data we used for our experiments in a csv file. 
		The CSV file contains the data in the following format: [date, text, crisis_type, event, time_series, sudden, utility, urgency, humanitarian]

4.Utils : 
	It contains side information we used for the alignement, in a repository and two file :
	4.1 crisis_knowledge_fr.csv
		This file is a csv in the following format : [Crisis_Type, Places, Crisis Name, Related station, label]
		Crisis_Type indicate what type of crisis the crisis belong (Hurricane, Storms ...)
		Places indicate where the crisis happend
		Crisis Name is the name of the crisis in the NLP data
		Related Station is the ID of the meteorological station of the administrative location the can be found in Time series data
		label indicate if the crisis is Sudden or Not Sudden (ecological)
		
	4.2 crisis_knowledge_eng.csv
		This file is a csv in the following format : [Crisis, Places, Path_Name]
		Crisis indicate what type of crisis the crisis belong (Hurricane, Storms ...)
		Places indicate where the crisis happend
		Path_Name is the name of the crisis in the NLP data
	
	4.3 Keywords :
		Is a repository that contains all the keywords we used for location mention detection.
		All the file are in the folloing format 'STATE_keywords_[city/county/state]_no_dupe.txt
		where a file contains all the keywords for a STATE regarding the following geographic information (city, county, state)
		

