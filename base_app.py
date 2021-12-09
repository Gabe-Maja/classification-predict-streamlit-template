"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
#import tweepy
#from textblob import TextBlob
#from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
#import matplotlib.pyplot as plt
from PIL import Image
#import seaborn as sns
import joblib,os
import string
from os import path


# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/raw_st.csv")
# Load your clean data
clean_csv = pd.read_csv("resources/clean_st.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Climate Change Belief Classifier")
	# st.markdown("<h1 style='text-align: center; color: black; font-size:54px;'>Climate Change Tweet Classifier</h1>", unsafe_allow_html=True)
	
	#st.image('https://www.tweetbinder.com/blog/wp-content/uploads/2018/07/classify-tweets-1.jpg', width=600,)

	st.image('https://i.imgur.com/W8NadTu.png')
	st.image('https://i.imgur.com/nAHbjxd.png')

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	menu = ["Homepage", "Data", "Visuals", "About"]
	selection = st.sidebar.image(
    "https://i.imgur.com/ylPx0KA.png",
    width=300,
)
	selection = st.sidebar.selectbox("", menu)	
	

	# Building out the Homepage page
	if selection == "Homepage":
		#st.subheader("Some background info about our classifier...")
		#st.info("Prediction with ML Models")
		#st.info("Believe it or not, but our App allows you to predict whether or not a person believes in climate change -- simply based off their tweets!")
		# Creating a text box for user input
		#st.image('https://i.imgur.com/hUwPTg2.png')
		tweet_text_raw = st.text_area('',"Enter Tweet Here")

		#clean the raw text (below)

		#pre-processing funtion no.1
		def pre_clean(tweet):
			tweet = re.sub(r'@[A-za-z0-9_]+', '', tweet) # remove twitter handles (@user)
			tweet = re.sub(r'https?:\/\/[A-za-z0-9\.\/]+', '', tweet) # remove http links
			tweet = re.sub(r'RT ', '', tweet) # remove 'RT'
			tweet = re.sub(r'[^a-zA-Z0-9 ]', '', tweet) # remove special characters, numbers and punctuations
			tweet = re.sub(r'#', '', tweet) # remove hashtag sign but keep the text
			tweet = tweet.lower() # transform to lowercase 
			return tweet


		#apply function no.1 to raw text
		tweet_text_clean = pre_clean(tweet_text_raw)


		model_options = ["Logistic Regression (base model)", "Stacking Classifier (optimal model)"]
		model_options_selection = st.selectbox("Choose a model:",
		model_options)

		pred_dic = {
			'[2]': 'News! (your tweet links to factual news about climate change)',
			'[1]': 'Pro! (your tweet supports the belief of man-made climate change)',
			'[0]': 'Neutral! (your tweet neither supports nor refutes the belief of man-made climate change)',
			'[-1]': 'Ant! (your tweet does not believe in man-made climate change)'
		}

		if model_options_selection == "Logistic Regression (base model)":		
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text_clean]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.

				#convert prediction from array to string
				prediction_str = np.array_str(prediction)

				st.success("Text Categorized as: {}".format(pred_dic[prediction_str]))

		if model_options_selection == "Stacking Classifier (optimal model)":		
			if st.button("Classify"):
				# Transforming user input with vectorizer
				#vect_text = tweet_cv.transform([tweet_text_clean]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				list_text_clean = tweet_text_clean.split() # convert string to list
				arr_text_clean = np.array(list_text_clean) # convert list to array
				predictor = joblib.load(open(os.path.join("resources/stack_class.pkl"),"rb"))
				prediction = predictor.predict([tweet_text_clean])

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.

				#convert prediction from array to string
				prediction_str = np.array_str(prediction)

				st.success("Text Categorized as: {}".format(pred_dic[prediction_str]))


	# Building out the "Data" page
	if selection == "Data":

		#st.image('https://i.imgur.com/KrEjOkC.png')

		st.info("Kindly navigate below to view the data files (tabular form).")
		# You can read a markdown file from supporting resources folder


		st.subheader("Twitter Data")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		if st.checkbox('Show clean data'): # data is hidden if box is unchecked
			st.write(clean_csv) # will write the df to the page
		st.subheader("")
		st.subheader("Sentiment Key")
		st.write("2: News  --  the tweet links to factual news about climate change")
		st.write("1: Pro  --  the tweet supports the belief of man-made climate change")
		st.write("0: Neutral  --  the tweet neither supports nor refutes the belief of man-made climate change")
		st.write("-1: Anti  --  the tweet does not believe in man-made climate change")


	# Building out the Visuals page
	if selection == "Visuals":
		#st.image('https://i.imgur.com/TbqaCEm.png')
		#st.sidebar.subheader("Let's visualise the data!")


		visual_options = ["Visuals (Home)", "Bar Graphs", "Word Clouds"]
		visual_options_selection = st.selectbox("Which visual category would you like?",
		visual_options)

		if visual_options_selection == "Visuals (Home)":
			st.image('https://i.imgur.com/7LeTUDC.png', width=730)


		if visual_options_selection == "Bar Graphs":
			st.image('https://i.imgur.com/p3J5Gcw.png')

			bar_nav_list = ['Sentiment distribution of raw data', 
			'Most common words in various sentiment classes (raw data)', 
			'Most common words in various sentiment classes (cleaned data)']

			bar_nav = st.selectbox('I would like to view the...', bar_nav_list)


			if bar_nav == 'Sentiment distribution of raw data':
				st.subheader('Sentiment Distribution of Raw Data')
				st.image('https://i.imgur.com/JT9HzVW.png', width=700)
				st.write("This graph shows how the raw data was distruted amongst the various sentiment classes.")
				st.write("The classes can be interpreted as follows:")				
				st.write("2: News  --  the tweet links to factual news about climate change")
				st.write("1: Pro  --  the tweet supports the belief of man-made climate change")
				st.write("0: Neutral  --  the tweet neither supports nor refutes the belief of man-made climate change")
				st.write("-1: Anti  --  the tweet does not believe in man-made climate change")
		
			if bar_nav == 'Most common words in various sentiment classes (raw data)':

				raw_common_words_list = ['All tweets', 'Negative tweets', 'Positive tweets', 
				'News-related tweets', 'Neutral tweets']
				raw_common_words = st.radio('Raw Data Sentiment Classes:', raw_common_words_list)

				if raw_common_words == 'All tweets':
					st.subheader('Common Words in all Tweets')
					st.image('https://i.imgur.com/HSdxDP9.png', width=700)

				if raw_common_words == 'Negative tweets':
					st.subheader('Common Words in Negative Tweets')
					st.image('https://i.imgur.com/FqnrN1Y.png', width=700)

				if raw_common_words == 'Positive tweets':
					st.subheader('Common Words in Positive Tweets')
					st.image('https://i.imgur.com/glF9Z0M.png', width=700)

				if raw_common_words == 'News-related tweets':
					st.subheader('Common Words in News-related Tweets')
					st.image('https://i.imgur.com/fWDhrTL.png', width=700)

				if raw_common_words == 'Neutral tweets':
					st.subheader('Common Words in Neutral Tweets')
					st.image('https://i.imgur.com/LEnkE9V.png', width=700)


			if bar_nav == 'Most common words in various sentiment classes (cleaned data)':

				clean_common_words_list = ['All tweets', 'Negative tweets', 'Positive tweets', 
				'News-related tweets', 'Neutral tweets']
				clean_common_words = st.radio('Cleaned Data Sentiment Classes:', clean_common_words_list)

				if clean_common_words == 'All tweets':
					st.subheader('Common Words in all Tweets')
					st.image('https://i.imgur.com/1aOr9DD.png', width=700)

				if clean_common_words == 'Negative tweets':
					st.subheader('Common Words in Negative Tweets')
					st.image('https://i.imgur.com/7Mp2NRX.png', width=700)

				if clean_common_words == 'Positive tweets':
					st.subheader('Common Words in Positive Tweets')
					st.image('https://i.imgur.com/3SDTh6c.png', width=700)

				if clean_common_words == 'News-related tweets':
					st.subheader('Common Words in News-related Tweets')
					st.image('https://i.imgur.com/sDFn7PF.png', width=700)

				if clean_common_words == 'Neutral tweets':
					st.subheader('Common Words in Neutral Tweets')
					st.image('https://i.imgur.com/0ur3J3C.png', width=700)

		if visual_options_selection == "Word Clouds":
			st.image('https://i.imgur.com/QDhrJTR.png')

			wc_nav_list = ['Most Common Words (Raw Data)', 
			'Most Common Words (Cleaned Data)']

			wc_nav = st.selectbox('I would like to view the...', wc_nav_list)


			if wc_nav == 'Most Common Words (Raw Data)':
				st.subheader('Most Common Words for all Tweets (raw data)')
				st.image('https://i.imgur.com/MsGuWFv.png')

			if wc_nav == 'Most Common Words (Cleaned Data)':

				wc_clean_list = ['All tweets', 'Negative tweets', 'Positive tweets', 
				'News-related tweets', 'Neutral tweets']
				wc_clean = st.radio('Cleaned Data Sentiment Classes:', wc_clean_list)

				if wc_clean == 'All tweets':
					st.subheader('Common Words in all Tweets')
					st.image('https://i.imgur.com/0MeELLk.png')

				if wc_clean == 'Negative tweets':
					st.subheader('Common Words in Negative Tweets')
					st.image('https://i.imgur.com/fc0Aa9l.png')

				if wc_clean == 'Positive tweets':
					st.subheader('Common Words in Positive Tweets')
					st.image('https://i.imgur.com/4LSqpBm.png')

				if wc_clean == 'News-related tweets':
					st.subheader('Common Words in News-related Tweets')
					st.image('https://i.imgur.com/cmrDvhk.png')

				if wc_clean == 'Neutral tweets':
					st.subheader('Common Words in Neutral Tweets')
					st.image('https://i.imgur.com/AJMYNOu.png')

	# Building out the "About" page 
	if selection == "About":
		#st.image('https://i.imgur.com/HHcTKkw.png')
		st.subheader("Some background info about our classifier...")
		st.info("Believe it or not, but our App allows you to predict whether or not a person believes in climate change -- simply based off their tweets!")
		
		st.markdown("The diagram below highlights the general process undergone to make all this possible.")
		st.image('https://i.imgur.com/lYk7iV2.png', width=700)
		st.subheader("")
		st.subheader("Our Chosen Models")

		model_list = ['<select model here>', 'Logistic Regression Model', 
		'Stacking Classifier Model']

		model_list_selection = st.selectbox('Which model would you like information on?', model_list)

		if model_list_selection == '<select model here>':
			st.markdown("")

		if model_list_selection == 'Logistic Regression Model':
			st.info('Logistic Regression is used to estimate the probability of whether an instance belongs to a class or not. If the estimated probability is greater than threshold, then the model predicts that the instance belongs to that class — or else it predicts that it does not belong to the class.')
			st.markdown('The image below shows an illustration of this')
			st.image('https://i.imgur.com/3CVWejZ.png', width=700)


		if model_list_selection == 'Stacking Classifier Model':
			st.info('Stacking is an ensemble learning technique to combine multiple classification models via a meta-classifier. The individual classification models are trained based on the complete training set. The meta-classifier is then fitted based on the outputs of the individual classification models. The meta-classifier can either be trained on the predicted class labels or probabilities from the ensemble.')
			st.markdown('The image below shows an illustration of this.')
			st.image('https://i.imgur.com/oG6eEzx.png', width=700)

		#st.write("Pretty neat right? ..We thought so too 😎")


# REFERENCES
# https://www.excelr.com/blog/data-science/regression/understanding-logistic-regression-using-r
# http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/





# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
