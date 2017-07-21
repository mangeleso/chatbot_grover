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

class request:
	def __init__(self, text, intent, entities):
		self.text = text
		self.intent = intent
		self.entities = entities

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

	INTENTS = ['ShowPriceByProduct', 'ShowSimilarProducts', 'ShowProductsFromBrand']

	phones = ['iphone', 'iphone apple','iphone 7', 'iphone plus', 'iphone 7 plus', 'apple iphone', 'iphone 128gb', 'iphone 32gb', 'iphone plus 128gb',  
			 'galaxy', 'smartphone', 'samsung galaxy', 'galaxy s8', 'galaxy 64gb', 'galaxy s8+ 64gb', 's8', 's8+']

	drones = ['drone', 'bebop', 'bebop 2']

	gaming = ['vive', 'htc', 'vr', 'virtual reality', 'glasses', 'rift', 'oculus rift']

	computing = ['mac', 'macbook', 'macbook air', 'mac pro', 'macbook pro', 'macbook 8gb', 'macbook 516gb', 'macbook 512gb', 'macbook air 11', 'macbook air 13', 'macbook pro 13', 'macbook 4gb'
				'macbook 500gb', 'lenovo yoga', 'yoga', 'surface', 'convertible surface', 'convertible yoga', 'microsoft surface']

	wearables = ['watch', 'watch 38', 'watch 42', 'watch 42mm', 'watch 38mm', 'apple watch', 'watch apple', 'suunto watch', 'watch ambit', 'ambit', 'watch v800', 
				'v800', 'polar watch', 'watch asus', 'asus watch']

	smart = ['amazon alexa', 'alexa', 'alexa dot', 'alexa echo', 'tchibo', 'tchibo qbo', 'qbo', 'qbo milk', 'samsung robot', 'robot vacuum', 'vacuum', 
			'vacuum cleaner', 'powerbot', 'VR20J9020UR', 'VR20J9259U']

	brands = ['apple', 'samsung', 'parrot', 'htc', 'oculus', 'microsoft', 'lenovo', 'suunto', 'polar', 'asus', 'amazon', 'tchibo']

	ENTITIES = { 'phones':phones, 'drones':drones, 'gaming':gaming, 'computing':computing, 'wearables':wearables, 'smart':smart}

	


	all_entities = phones + drones + gaming + computing + wearables + smart

	show_price = ['show me price', 'how much', 'cost of', 'tell me the price', 'give me the price', 'how much money']
	show_products = ['show me', 'I want a', 'I need a', 'I wish a', 'I am looking for a', 'I look for a', 'show me a', 'I am interested','tell me about']
	show_products_brand = ['show', 'looking for', 'do you have', 'products']


	data_show_price = []
	for ent in all_entities:
		for s in show_price:
			data_show_price.append(s + " " + ent)
	labels_0 = [0] * len(data_show_price)

	data_show_similar_products = []
	for ent in all_entities:
		for s in show_products:
			data_show_similar_products.append(s + " " + ent)

	#Append all entities in the list 
	data_show_similar_products = data_show_similar_products + all_entities
	labels_1 = [1] * len(data_show_similar_products) 

	data_brands = []
	for b in brands:
		for s in show_products_brand:
			data_brands.append(s + " " + b)
	#Append all brands 
	data_brands = data_brands + brands
	labels_2 = [2] * len(data_brands)
			
	return data_show_price + data_show_similar_products + data_brands, labels_0 + labels_1 + labels_2


def find_intent(clf, input_sentence, transformer):
	'''
	Funtion that maps a input sentence from the user to a intent 
	0: Search Price of products 
	1: Search similar products
	2: Search products by brand 
	'''
	#Preprocess the input string
	clean_sentence = preprocess_senteces(input_sentence)

	#Create the features with the transformer
	features = transformer.transform([clean_sentence])

	#If the prediction is not so sure from the user input return -1
	proba = clf.predict_proba(features)[0]

	proba = np.array(proba)

	if (proba > 0.7).sum() > 0:
		return clf.predict(features)[0]

	return -1

def find_entities(sentence):
	entities = []
	clean_s = preprocess_senteces(sentence)

	# function to test if something is a noun
	is_noun = lambda pos: pos[:2] == 'NN'

	# do the nlp stuff
	tokenized = nltk.word_tokenize(clean_s)
	nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
	return nouns


def do_intent(intent, entities, df):
	'''
	Execute the different intents:
	0: Search Price of products 
	1: Search similar products
	2: Search products by brand 
	'''
	#Make sure entities is no empty
	if not entities:
		#print("Bot: I am sorry but at the moment we don't have what you are looking for :/...")
		return

	if intent == 1: 
		products = df.loc[df['Product Name'].str.contains('|'.join(map(re.escape, entities)), case=False)]	
		if products.empty:
			return

		print("Bot: We have the following products you might be interested:")
		for prod in products['Product Name'].values:
			print("Bot: " + prod)
		return products

	elif intent == 0:
		products = df.loc[df['Product Name'].str.contains('|'.join(map(re.escape, entities)), case=False)]
		print("Bot: We have the following prices for the products searched:")
		for prod,price  in products[['Product Name', 'Subscription Plan']].values:
			print("Bot: The montly price for " + prod + " is: " + str(price) + " .")

	elif intent == 2:
		products = df.loc[df['Brand'].str.contains('|'.join(map(re.escape, entities)), case=False)]	
		if products.empty:
			return

		print("Bot: We have the following products from " + entities[0])
		for prod in products['Product Name'].values:
			print("Bot: " + prod)
		return products

def is_a_brand(brand_list, brand):
	for b in brand_list:
		if brand == b.lower():
			return True 
	return False

def search_price_print(product, df):
	'''
		Df is a dataframe with only products similar to the users choice
	'''
	one_product = df.loc[df['Product Name'].str.contains(product, case=False, regex=False)]

	if one_product.empty:
		print("Bot: Sorry, but this product is not available...")

	else:
		for name, price in one_product[['Product Name', 'Subscription Plan']].values:
			print("Bot: The montly price for " + name + " is: " + str(price) + " .")

	
def print_error_msg():
	print("Bot: I am affraid we dont have what are you looking for..:/, please try again.")


if __name__ == '__main__':
	word_lemm = WordNetLemmatizer()
	
	#Load dataframe 
	df = pd.read_csv('data.csv', index_col=0)
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
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, stop_words = None, max_features = 100)
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
	forest2 = RandomForestClassifier(n_estimators=50)
	forest2 = forest2.fit(train_tfidf, labels)

	#Train a nearest neighbor classifier
	neigh = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
	neigh.fit(train_data_features, labels)

	#Some test sentences
	test_sentences = ['I want to know more about an iphone', 'I am interested in a s8', 'show me price for alexa echo', 'I look for shoes', 'apple iphone', 'show samsung', 'samsung']

	clean_test = [preprocess_senteces(x) for x in test_sentences]
	test_features = vectorizer.transform(clean_test)
	test_tfidf = vectorizer_tfidf.transform(clean_test)

	#print(forest.predict(test_features))

	#Visualize some training data using TSNE
	visualize = TSNE(n_components=2, random_state=42)
	visualize_data = visualize.fit_transform(train_tfidf)
	x_v = visualize_data[:,0]
	y_v = visualize_data[:,1]
	plt.scatter(x_v, y_v, c=labels)
	#plt.show()

	requests = []
	cats = pd.unique(df['Category'])
	no_cats = len(cats)
	INITIAL_GREETING = "Bot: Welcome at Grover!!, I am your personal assistant"

	print(INITIAL_GREETING)
	#Main loop 
	try: 
		while True:
			print("")
			print('Bot: What are you looking for?')
			user_input = input('User: ')
			intent = find_intent(forest, user_input, vectorizer)
			entities = find_entities(user_input)
			requests.append(request(user_input, intent, entities)) 

			if intent < 0:
				print_error_msg()

			elif intent == 0:
				do_intent(intent, entities, df)
			
			elif intent >= 1:
				products = do_intent(intent, entities, df)
				if products is None:
					print_error_msg()

				else:
					print("Bot: Are you interested in any of the products y/n?")
					res = input("User: ")
					if 'y' in res:
						print("Bot: Which characteristics?")
						res = input("User: ")
						price = search_price_print(res, products)
	
	except KeyboardInterrupt:
		print('Bot: Have a nice day!')

	
