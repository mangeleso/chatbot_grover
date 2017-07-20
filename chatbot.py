# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
from sklearn import decomposition


'''
We will use a retrieval-based model using a RandomFores and KNN classifier, the system dont generate new text, 
it just pick a response from a fixed set. The user is able to type anything he is interested, if products available
the Assistant will display the brands and later the products and price from the input of the user.
'''


def preprocess_senteces( raw_sentences):
	'''
	Function that converts a raw request to a string of words
	The input is a string (request from training set)
	The output is a string (preprocessed request)
	'''

	#Remove non-letters
	letters_only = re.sub("[^a-zA-Z0-9]", " ", raw_sentences) 

	#Convert to lower_case
	words = letters_only.lower().split()

	stops = set(stopwords.words("english"))

	#Remove stop words
	word_lemm = WordNetLemmatizer()
	meaningful_words = [word_lemm.lemmatize(w, pos='v') for w in words if not w in stops]

	#Join the words back into one string 
	return(" ".join(meaningful_words))



def create_data():
	'''
	This is an example to create some data in order to train the machine learning model
	'''
	WORDS_DESIRE = ['I want a', 'I need a', 'I wish a', 'I am looking for a', 'I look for a', 'show me a', 'I am interested']
	samsung = ['samsung', 'galaxy', 'smartphone', 'samsung galaxy', 'galaxy s8', 'phone']
	iphone = ['iphone', 'iphone 7', 'plus', 'apple iphone']
	phones = samsung + iphone
	drone = ['drone', 'parrot', 'plane']
	gaming = ['vive', 'htc', 'virtual reality', 'virtual glasses', 'vr', 'oculus', 'oculus rift']
	computing = ['apple macbook', 'apple macbook air', 'mac', 'laptop', 'computer']
	wearables = ['apple watch', 'smartwatch' , 'watch', 'ambit', 'suunto', 'polar', 'v800']
	smart_home = ['alexa dot', 'alexa echo', 'amazon', 'qbo milk master' , 'tchibo', 'robotic vacuum cleaner']
	all_cats = [phones, drone, gaming, computing, wearables, smart_home]

	sentences = []
	labels = []

	for idx, cat in enumerate(all_cats):
		for product in cat:
			for desire in WORDS_DESIRE:
				sentences.append(desire + " " + product)
		labels = labels + [idx] * (len(cat) * len(WORDS_DESIRE))
	return sentences, labels 


def find_category(clf, input_sentence, transformer):
	'''
	Funtion that maps a input sentence from the user to the 
	desire category from {Phones: 0, Drones: 1, Gaming:2, Computing: 3, Wearables:4 , Smarthomes:5}
	It returns the category number 
	'''
	#Preprocess the input string
	clean_sentence = preprocess_senteces(input_sentence)

	#Create the features with the transformer
	features = transformer.transform([clean_sentence])

	#If the prediction is not so sure from the user input return -1
	proba = clf.predict_proba(features)[0]
	#print(proba)
	proba = np.array(proba)

	if (proba > 0.5).sum() > 0:
		return clf.predict(features)[0]

	return -1


def display_brands(cat, df):
	'''
		CAT = {Phones: 0, Drones: 1, Gaming:2, Computing: 3, Wearables:4 , Smarthomes:5}
	'''
	#Get products from category and display
	cats = {0:'Phones & Tablets', 1:'Drones', 2:'Gaming & VR', 3:'Computing', 4:'Wearables', 5:'Smart Home'}

	brands = df['Brand'].loc[df['Category'] == cats[cat]].values

	brands = set(brands)

	print("I see, you are interested in " + cats[cat] + "!")
	print("We currently have the following brands: ")

	for idx, brand in enumerate(brands):
		print(brand)
	
	return brands


def is_a_brand(brand_list, brand):
	for b in brand_list:
		if brand == b.lower():
			return True 
	return False


def display_products(brand, df, cat):

	cats = {0:'Phones & Tablets', 1:'Drones', 2:'Gaming & VR', 3:'Computing', 4:'Wearables', 5:'Smart Home'}

	all_brands = df['Brand'].values
	products = []
	for b in all_brands:
		if brand == b.lower():
			mask = (df['Brand'] == b ) & (df['Category'] == cats[cat])
			products = df['Product Name'].loc[ mask ].values
			print("Here are all products from " + b + "!")
			break

	for idx, product in enumerate(products):
		print( str(idx) + ": " + product)

	return products


	
def display_subscription(index, products, df):
	if index >= len(products):
		print("Incorrect index")
 
	else:
		price = df['Subscription Plan'].loc[df['Product Name'] == products[index]].values[0]
		print("The price for " + products[index] + " is: " + str(price) + " for a month.")


if __name__ == '__main__':
	word_lemm = WordNetLemmatizer()
	
	#Load dataframe 
	df = pd.read_csv('data.csv', index_col=0)
	print(df.head())


	#Create some data 
	sentences, labels = create_data()

	clean_sentences = []

	#Clean the training data 
	for s in sentences:
		clean_sentences.append(preprocess_senteces(s))

	#print(clean_sentences)

	#Split train test 
	X_train, X_test, y_train, y_test = train_test_split(clean_sentences, labels, test_size=0.2, random_state=42)

	#Create features -  Bag of words
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, stop_words = None, max_features = 100)
	vectorizer_tfidf = TfidfVectorizer(use_idf = False)

	#Bag of words model
	train_data_features = vectorizer.fit_transform(clean_sentences)
	train_data_features = train_data_features.toarray()
	print(train_data_features.shape)

	#Bag of TF-IDF model
	train_tfidf = vectorizer_tfidf.fit_transform(clean_sentences)
	train_tfidf = train_tfidf.toarray()

	#Train a classifier
	forest = RandomForestClassifier(n_estimators = 40)
	forest = forest.fit(train_data_features, labels)

	#Train a randomforest using td-idf features
	forest2 = RandomForestClassifier(n_estimators=40)
	forest2 = forest2.fit(train_tfidf, labels)

	#Train a nearest neighbor classifier
	neigh = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
	neigh.fit(train_data_features, labels)


	#Some test sentences
	test_sentences = ['I want to know more about an iphone', 'I am interested in a s8', 'show me price for alexa echo', 'I look for shoes']
	clean_test = [preprocess_senteces(x) for x in test_sentences]
	test_features = vectorizer.transform(clean_test)
	test_tfidf = vectorizer_tfidf.transform(clean_test)

	print(forest.predict(test_features))

	#Visualize some training data using TSNE
	visualize = TSNE(n_components=2, random_state=42)
	visualize_data = visualize.fit_transform(train_tfidf)

	print(visualize_data.shape)
	x_v = visualize_data[:,0]
	y_v = visualize_data[:,1]
	plt.scatter(x_v, y_v, c=labels)
	#plt.show()


	cats = pd.unique(df['Category'])
	no_cats = len(cats)
	INITIAL_GREETING = "Welcome at Grover!!, I am your personal assistant"

	print(INITIAL_GREETING)

	try: 
		while True:
			user_input = input('What are you looking for?\n')
			cat = find_category(forest, user_input, vectorizer)
			
			if cat < 0:
				print("I am affraid we dont have what are you looking for..:/, please try again.")

			else:#Print the brands from the data

				brands_interest = display_brands(cat, df)

				#Ask for the brand of interest 
				brand = preprocess_senteces(input("From which brand you want to see products?"))
				if is_a_brand(brands_interest, brand):
					#Show products from the brand
					products = display_products(brand, df, cat)

					product_user = input("Please type the id of the product :).")

					#Display the plab subscription
					display_subscription(int(product_user), products, df)


	except KeyboardInterrupt:
		print('Have a nice day!')





	#Create test features
	#test_data_features = vectorizer.fit_transform(clean_sentences)
	#test_data_features = test_data_features.toarray()

	#With tfidf features
	#test_tfidf = vectorizer_tfidf.fit_transform(clean_sentences)


	print(forest.predict(test_features))
	print(forest2.predict(test_tfidf))
	print(neigh.predict(test_features))

	#Print the score for the two Random forest, each trained with different features
	
	#print(forest.score(train_data_features, labels))
	#print(forest2.score(train_data_features , labels))







'''
Easy harcoded implementation 

try: 
	while True:
		#Print categories available
		print("What categories are you interested in (type the id):")
		for j in range(len(cats)):
			print(str(j) + ": " + cats[j])
	
		user_category = int(input())

		if user_category >= len(cats):
			print("This option doesnt exist")

		#Get products from category and display
		products = df['Product Name'].loc[df['Category'] == cats[user_category]].values

		print("Products available from " + cats[user_category] + " found:")
		for i in range(len(products)):
			print(str(i) + ": " + products[i])

		#Get the interested product:
		id_product = int(input())

		#Display the plan subscription
		plans = df['Subscription Plan'].loc[df['Product Name'] == products[id_product]].values
		price_plan =  str(df['Subscription Plan'].loc[df['Product Name'] == products[id_product]].values[0])
		print("Price subscription: " + price_plan) 
		if input("Do you want to keep searching for more products y/n? \n") == 'n':
			print("Have a nice day!")
			break

except KeyboardInterrupt:
	print('Interrupted!')

'''
	