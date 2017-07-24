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
from nltk.corpus import wordnet as wn
from textblob import TextBlob  
from nltk.stem.lancaster import LancasterStemmer

'''
Conversation application to retrieve information from products and prices. The Chatbot is trained to distinguish 3 intents:
- Show Producs by name 
- Show products by brand
- Show products by object type

The chatbot is trained on a RandomForest using the bag of words model. 

'''

class request: 
	def __init__(self, text, intent, entities):
		self.text = text
		self.intent = intent
		self.entities = entities

def display_help():
	print("Just type any product, brand or gadget you are interested :), I will do my best to find it!")


def preprocess_senteces( raw_sentences):
	'''
	Function that converts a raw request to a string of words
	The input is a string (request from training set)
	The output is a string (preprocessed request)
	'''
	#Remove non-letters
	letters_only = re.sub("[^a-zA-Z0-9+]", " ", raw_sentences) 

	#Convert to lower_case
	words = letters_only.lower().split()

	stops = set(stopwords.words("english"))

	#Remove stop words
	word_lemm = WordNetLemmatizer()
	meaningful_words = [word_lemm.lemmatize(w, pos='v') for w in words if not w in stops]

	#Join the words back into one string 
	return(" ".join(meaningful_words))


def more_data():
	'''
	This is an example to create some data in order to train the machine learning model to classify the intents 
	'''
	#ProductsName
	phones = ['iphone', 'iphone apple','iphone 7', 'iphone plus', 'iphone 7 plus', 'apple iphone', 'iphone 128gb', 'iphone 32gb', 'iphone plus 128gb',  
			 'galaxy', 'samsung galaxy', 'galaxy s8', 'galaxy 64gb', 'galaxy s8+ 64gb', 's8', 's8+']
	drones = [ 'bebop', 'bebop 2']
	gaming = ['vive', 'glasses rift', 'rift', 'oculus rift', 'rift vr']
	computing = ['mac', 'macbook', 'macbook air', 'mac pro', 'macbook pro', 'macbook 8gb', 'macbook 516gb', 'macbook 512gb', 'macbook air 11', 'macbook air 13', 'macbook pro 13', 'macbook 4gb'
				'macbook 500gb', 'lenovo yoga', 'yoga', 'surface', 'convertible surface', 'convertible yoga', 'microsoft surface']
	wearables = ['watch 38', 'watch 42', 'watch 42mm', 'watch 38mm', 'apple watch', 'watch apple', 'suunto watch', 'watch ambit', 'ambit', 'watch v800', 
				'v800', 'polar watch', 'watch asus', 'asus watch']
	smart = ['amazon alexa', 'alexa', 'alexa dot', 'alexa echo', 'tchibo', 'tchibo qbo', 'qbo', 'qbo milk', 'samsung robot',   
			'vacuum cleaner', 'powerbot', 'VR20J9020UR', 'VR20J9259U']
	products_names = phones + drones + gaming + computing + wearables + smart

	#Brands
	product_brands = ['apple', 'samsung', 'parrot', 'htc', 'oculus', 'microsoft', 'lenovo', 'suunto', 'polar', 'asus', 'amazon', 'tchibo']

	#Nouns 	
	phone =  ['phone','smart phone','telephone','cellphone','smartphone','mobile','mobile phone','mobilephone', 'smartphones', 'mobiles']
	drone = ['drone','plane', 'planes', 'drones']
	games = ['game','vr','virtual reality', 'games', 'gaming', 'virtual', 'reality']
	computing = ['computer','laptop', 'computers', 'laptops']
	wearables = ['wearable','smart watch','watch', 'watches', 'smart watches']
	assistant = ['personal assistant','intelligent personal assistant','smart assistant','smart personal assistant']
	coffee =['coffee','coffee machine','smart coffe','smart coffee machine']
	vacuum = ['vacuum','vacuum cleaner','smart vacuum']
	product_nouns = phone + drone + games + computing + wearables +  assistant + coffee +  vacuum

	#Requests 
	show_price = ['show me price', 'how much', 'cost of', 'tell me the price', 'give me the price', 'how much money','price of']
	show_me = ['show me', 'I want a', 'I need a', 'I wish a', 'I am looking for a', 'I look for a', 'show me a', 'I am interested','tell me about']
	dont_like = ['i dont like', 'dont like', 'i hate', 'i do not like']


	#Generate dataset to train classifier of intents

	#Intent 0  - Show price of product
	data_price = []
	for product in products_names:
		for s in show_price:
			data_price.append(s + " " + product)
	labels_0 = [0] * len(data_price)

	#Intent 1 - Show products by name
	data_names = []
	for product in products_names:
		for s in show_me:
			data_names.append(s + " " + product)

	data_names = data_names + products_names
	labels_1 = [1] * len(data_names) 

	#Intent 2 - Show by brand 
	data_brands = []
	for brand in product_brands:
		for s in show_me:
			data_brands.append(s + " " + brand)
	data_brands = data_brands + product_brands
	labels_2 = [2] * len(data_brands)
	
	#Intent 3 - Show by nouns
	data_nouns = []
	for noun in product_nouns:
		for s in show_me:
			data_nouns.append(s + " " + noun)
	data_nouns = data_nouns + product_nouns
	labels_3 = [3] * len(data_nouns)

	return  data_names + data_brands + data_nouns ,  labels_1 + labels_2 + labels_3

def append_synonyms(df):
	'''
	Appends a new column into the data for the nouns and synonyms
	'''
	#Nouns 
	phone =  ['phone','smart phone','telephone','cellphone','smartphone','mobile','mobile phone','mobilephone', 'smartphones', 'mobiles']
	drone = ['drone','plane', 'planes', 'drones']
	games = ['game','vr','virtual reality', 'games', 'gaming', 'virtual', 'reality']
	computing = ['computer','laptop', 'computers', 'laptops']
	wearables = ['wearable','smart watch','watch', 'watches', 'smart watches']
	assistant = ['personal assistant','intelligent personal assistant','smart assistant','smart personal assistant']
	coffee =['coffee','coffee machine','smart coffe','smart coffee machine']
	vacuum = ['vacuum','vacuum cleaner','smart vacuum']

	#Store as a string
	phone = ','.join(x for x in phone)
	drone = ','.join(x for x in drone)
	games = ','.join(x for x in games)
	computing = ','.join(x for  x in computing)
	wearables = ','.join(x for x in wearables)
	assistant = ','.join(x for x in assistant)
	coffee = ','.join(x for x in coffee)
	vacuum = ','.join(x for x in vacuum)

	new_column = [ phone , phone , phone , phone, phone, drone, drone, games, games, computing, computing, computing, computing, computing, computing, 
					wearables, wearables, wearables, wearables, wearables, assistant, assistant , coffee , vacuum, vacuum]

	df['Synonyms'] = new_column
	return df


def find_intent(clf, input_sentence, transformer):
	'''
	Funtion that maps a input sentence from the user to a intent 
	0: Search Price of products 
	1: Search similar products
	2: Search products by brand 
	'''

	#Create the features with the transformer
	features = transformer.transform([input_sentence])

	#If the prediction is not so sure from the user input return -1
	proba = clf.predict_proba(features)[0]

	proba = np.array(proba)
	#print(proba)

	if (proba > 0.5).sum() > 0:
		return clf.predict(features)[0]

	return -1

def find_entities(sentence, df):
	entities = []

	# function to test if something is a noun
	is_noun = lambda pos: pos[:2] == 'NN'

	# do the nlp stuff
	tokenized = nltk.word_tokenize(sentence)

	#Brands
	brands = df['Brand'].values
	brands = set(brands)
	brands = [x.lower() for x in brands]
	
	#Objects
	synonyms = df['Synonyms'].values
	synonyms = synonyms.tolist()
	new_synonyms = [x.split(',') for x in synonyms]
	flatten_synonyms = [item for sublist in new_synonyms for item in sublist]
	flatten_synonyms = set(flatten_synonyms)

	nouns = []
	for (word, pos) in nltk.pos_tag(tokenized):
		if word in brands:
			nouns.append(word)
		elif word in flatten_synonyms:
			nouns.append(word)
		elif is_noun(pos):
			nouns.append(word)

	#nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
	return nouns


def display_products(df, with_price=False):
	'''
	Display the products names found 
	'''
	if len(df)<1:

		return	

	if not with_price:
		print("\nBot: We have the following products you might be interested:\n")
		for prod in df['Product Name'].values:
			print("Bot: " + prod)

	else:
		print("Bot: Here are the montly price for the products: \n")
		for prod,price  in df[['Product Name', 'Subscription Plan']].values:
			print("Bot: The montly price for " + prod + " is: " + str(price) + " .")


def filter_brand_name(df, entities, intent):
	'''
	Filters one more time using the entities found and the type of intent. Displays the products found.
	'''
	#Filter some entities
	#if intent==1:

	#Brand
	if intent==2:
		#Remove Brands 
		brands = df['Brand'].values
		brands = set(brands)
		brands = [x.lower() for x in brands]
		new_entities = [x for x in entities if not x in brands]

		#Filter by object type
		if len(new_entities) > 0:
			products = df.loc[df['Synonyms'].str.contains('|'.join(map(re.escape, new_entities)), case=False)]
			if products.empty:
				products = df.loc[df['Product Name'].str.contains('|'.join(map(re.escape, new_entities)), case=False)]		
				if products.empty:
					return
			display_products(products)
			return products
		

	elif intent==3:
		#Remove synonyms from entities 
		synonyms = df['Synonyms'].values
		synonyms = synonyms.tolist()
		new_synonyms = [x.split(',') for x in synonyms]
		flatten_synonyms = [item for sublist in new_synonyms for item in sublist]
		flatten_synonyms = set(flatten_synonyms)
		new_entities = [x for x in entities if x not in flatten_synonyms]

		if len(new_entities) > 0:
			products = df.loc[df['Brand'].str.contains('|'.join(map(re.escape, new_entities)), case=False)]
			if products.empty:
				products = df.loc[df['Product Name'].str.contains('|'.join(map(re.escape, new_entities)), case=False)]
				if products.empty:
					return
				display_products(products)
				return products

			display_products(products)
			return products


	display_products(df)
	return df




def do_intent(intent, entities, df):
	'''
	Execute the different intents:
	1: Search by product name
	2: Search by brand 
	3: Search by object tyope 
	'''
	#Make sure entities is no empty
	if not entities:
		return

	# Search price 
	if intent == 0:
		products = df.loc[df['Product Name'].str.contains('|'.join(map(re.escape, entities)), case=False)]
		if products.empty:
			return None, None
		display_products(products)
		print("\nBot: Do you want to see the price of all y/n?")
		reply = input("User: ")
		if 'y' in reply:
			display_products(products, True)
			return products, False
		return products, True

	# Search by product names
	elif intent == 1: 
		products = df.loc[df['Product Name'].str.contains('|'.join(map(re.escape, entities)), case=False)]	
		if products.empty:
			return
		return filter_brand_name(products, entities, 1)

	#Search by brand
	elif intent == 2:
		products = df.loc[df['Brand'].str.contains('|'.join(map(re.escape, entities)), case=False)]	
		if products.empty:
			return
		return filter_brand_name(products, entities, 2)

	#Search by noun 
	elif intent == 3:
		products = df.loc[df['Synonyms'].str.contains('|'.join(map(re.escape, entities)), case=False)]	
		if products.empty:
			return
		return filter_brand_name(products, entities, 3)




def search_price_print(product, df):
	'''
		Df is a dataframe with only products similar to the users choice
	'''
	one_product = df.loc[df['Product Name'].str.contains(product, case=False, regex=False)]

	if one_product.empty:
		print("\nBot: Sorry, but this product is not available...")

	else:
		if len(one_product) > 1:
			print("Bot: We have the following products you might be interested:\n")
		else:
			print("Bot: We have the following product you might be interested:\n")
		for name, price in one_product[['Product Name', 'Subscription Plan']].values:
			print("Bot: The montly price for " + name + " is: " + str(price) + " .")


	
def print_error_msg():
	print("Bot: I am affraid we dont have what are you looking for..:/, please try again.")


if __name__ == '__main__':
	word_lemm = WordNetLemmatizer()
	
	#Load dataframe 
	data = pd.read_csv('data.csv', index_col=0)

	#Append synonyms into data
	df = append_synonyms(data)

	#print(df.head())

	#Create some data 
	#sentences, labels = create_data()
	sentences, labels = more_data()

	#Clean the training data 
	clean_sentences = []
	for s in sentences:
		clean_sentences.append(preprocess_senteces(s))

	#Split train test 
	X_train, X_test, y_train, y_test = train_test_split(clean_sentences, labels, test_size=0.2, random_state=42)

	#Create features -  Bag of words
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, stop_words = None, max_features = 200)
	vectorizer_tfidf = TfidfVectorizer(use_idf = False)

	#Bag of words model
	train_data_features = vectorizer.fit_transform(clean_sentences)
	train_data_features = train_data_features.toarray()
	#print(train_data_features.shape)

	#Bag of TF-IDF model
	train_tfidf = vectorizer_tfidf.fit_transform(clean_sentences)
	train_tfidf = train_tfidf.toarray()

	#Train a classifier
	forest = RandomForestClassifier(n_estimators=50)
	forest = forest.fit(train_data_features, labels)

	#Train a randomforest using td-idf features
	#forest2 = RandomForestClassifier(n_estimators=50)
	#forest2 = forest2.fit(train_tfidf, labels)

	#Some test sentences
	test_sentences = ['I want to know more about an iphone', 'I am interested in a s8', 'show me price for alexa echo', 'I look for shoes', 'apple iphone', 'show samsung' ]

	clean_test = [preprocess_senteces(x) for x in test_sentences]
	test_features = vectorizer.transform(clean_test)
	test_tfidf = vectorizer_tfidf.transform(clean_test)
	#print(forest.predict(test_features))


	requests = []
	cats = pd.unique(df['Category'])
	no_cats = len(cats)
	INITIAL_GREETING = "Bot: Welcome at Grover!!, I am your personal assistant"
	print(INITIAL_GREETING)
	#Main loop 
	try: 
		while True:
			print("")
			print('Bot: Type what are you looking for.')
			user_input = input('User: ')
			user_input = preprocess_senteces(user_input)

			if len(user_input) < 3:
				print_error_msg()

			elif user_input.lower() == 'help':
				display_help()
					
			else:
				intent = find_intent(forest, user_input, vectorizer)
				entities = find_entities(user_input, df)

				requests.append(request(user_input, intent, entities)) 

				if intent < 0:
					print_error_msg()
				
				else:
					products = do_intent(intent, entities, df)

					if products is None:
						print_error_msg()

					else:
						print("\nBot: Are you interested in any of the products y/n?")
						res = input("User: ")
						if 'y' in res:
							print("Bot: Could you specify which product are you interested?")
							res = input("User: ")
							price = search_price_print(res, products)

						else:
							print("Bot: I am sorry to hear that. You can search for a new product, I will try my best!")
			
	except KeyboardInterrupt:
		print('Bot: Have a nice day!')

	
