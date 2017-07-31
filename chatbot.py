# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from textblob import TextBlob  
import logging 
import random 
import os
from nltk import word_tokenize
from nltk.tag import StanfordNERTagger, StanfordPOSTagger
from textblob.en.np_extractors import ConllExtractor, FastNPExtractor


np.random.seed(42)

#Load csv into a pandas dataframe
df = pd.read_csv('data.csv', index_col=0)
df_last_display = pd.DataFrame()
last_type_of_question = 0

COLS = ['Product Id','Product Name', 'Brand', 'Category', 'Subscription Plan', 'Synonyms']


#Brands names
BRANDS = [x.lower() for x in list(set(df['Brand'].values))]

PRODUCT = []
for word in df['Product Name'].values:
	for w in word.split():
		if w not in PRODUCT:
			PRODUCT.append(w.lower())


#Type objects or synonyms 
phone =  ['phone', 'phones','smart phone', 'smart phones' , 'telephone','cellphone','smartphone','mobile','mobile phone','mobilephone', 'smartphones', 'mobiles']
phone_apple =  ['iOS', 'iphone', 'iphones'] + phone
phone_samsung =  ['android', 'galaxy'] + phone
drone = ['drone','plane', 'planes', 'drones']
games = ['game','vr','virtual reality', 'games', 'gaming', 'virtual', 'reality']
computing = ['computer','laptop', 'computers', 'laptops']
wearables = ['wearable','smart watch','watch', 'watches', 'smart watches', 'wearables']
assistant = ['personal assistant','intelligent personal assistant','smart assistant','smart personal assistant']
coffee =['coffee','coffee machine','smart coffe','smart coffee machine']
vacuum = ['vacuum','vacuum cleaner','smart vacuum']
PRODUCTS_TYPES = phone + drone + games + computing + wearables + assistant + coffee + vacuum + ['ios', 'android' , 'iphones', 'galaxy']
YES = ['yes', 'yea',"affirmative", 'yey', 'yeah', 'alright', 'of course', 'sure', 'ok', 'okey', 'agreed', 'certainly', 'oki', 'absolutely', 'yay', 'for sure']

def add_synonyms():
	'''Append a columns called 'Synonyms' that stores the type of the product and synonyms of it.
	'''
	global df

	#Store as a string
	s_phone_a = ','.join(x for x in phone_apple)
	s_phone_s = ','.join(x for x in phone_samsung)
	s_drone = ','.join(x for x in drone)
	s_games = ','.join(x for x in games)
	s_computing = ','.join(x for  x in computing)
	s_wearables = ','.join(x for x in wearables)
	s_assistant = ','.join(x for x in assistant)
	s_coffee = ','.join(x for x in coffee)
	s_vacuum = ','.join(x for x in vacuum)
	new_column = [ s_phone_a , s_phone_a , s_phone_a , s_phone_s, s_phone_s , s_drone, s_drone, s_games, s_games, s_computing, s_computing, s_computing, s_computing, s_computing, s_computing, 
					s_wearables, s_wearables, s_wearables, s_wearables, s_wearables, s_assistant, s_assistant , s_coffee , s_vacuum, s_vacuum]
	df['Synonyms'] = new_column



def create_data():
	'''
	This is an example to create some data in order to train the machine learning model to classify the intents 
	'''
	#ProductsName
	products_names = PRODUCT

	#Brands
	#product_brands = ['apple', 'samsung', 'parrot', 'htc', 'oculus', 'microsoft', 'lenovo', 'suunto', 'polar', 'asus', 'amazon', 'tchibo']
	product_brands = BRANDS

	#Nouns 	
	product_nouns = PRODUCTS_TYPES

	#Requests 
	show_price = ['show me price', 'how much', 'cost of', 'tell me the price', 'give me the price', 'how much money','price of']
	show_me = ['show me', 'I want a', 'I need a', 'I wish a', 'I am looking for a', 'I look for a', 'show me a', 'I am interested','tell me about' ]
	dont_like = ["I don't like", 'I hate', "I am not interested", "I have no interest"]
	but_like = ['I like', 'show me something different', 'I prefer', 'I want']

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

	#Infer from a dislike (brand|product name|)
	data_dislike = []
	all_prod = product_brands + products_names
	for product in all_prod:
		for s in range(len(dont_like)):
			data_dislike.append(dont_like[s] + " " + product)
			data_dislike.append(dont_like[s] + " " + product + " " +but_like[s])


	labels_4 = [4] * len(data_dislike)

	return  data_names + data_brands + data_nouns + data_dislike,  labels_1 + labels_2 + labels_3 + labels_4



def display_help():
	print("Just type any product, brand or gadget you are interested :), I will do my best to find it!")

def print_error_msg():
	print("Bot: I am affraid we dont have what are you looking for..:/, please try again.")

def not_found():
	print("Bot: I couldn't find a good match for your search, please try again")	

def print_not_understand():
	print("Bot: I am sorry, but I couldn't understand you.")

def preprocess_text(sentence):
	'''Remove weird symbols and non alphanumeric, sign '+' stays,
	 	also need to capitalize 'i'
	'''
	#Keep don't  instead of dont or do not
	sentence = sentence.replace("dont", "don't")
	sentence = sentence.replace("do not", "don't")
	sentence = sentence.replace("doesnt", "doesn't")
	sentence = sentence.replace("does not", "doesn't")

	#Remove non-letters, non-numbers, non-+s
	letters_only = re.sub("[^a-zA-Z0-9+']", " ", sentence) 

	cleaned = []
	#Convert to lower_case
	words = letters_only.lower().split()

	for w in words:
		if w == 'i':
			w = 'I'
		if w == "i'm":
			w = "I'm"
		cleaned.append(w)
	return ' '.join(cleaned)


'''
def find_pronoun(sent):
	#Finds the pronoun to respond with, returns  none if not pronoun was found
	
	pronoun = None

	for word, pos in sent.pos_tags:
		#Disambiguate pronouns
		if pos == 'PRP' and word.lower() == 'you':
			pronoun = 'I'
		elif pos == 'PRP' and word == 'I':
			pronoun = 'You'
	return pronoun
'''

def convert_singular(w):
	'''Simple huristic to convert a noun into a singular form
	'''
	if w == 'iphones':
		return 'iphone'
	elif w[-1] == 's':
		if w[-2] in ('a', 'e', 'i', 'o', 'u'): # if second last is vowel watches -> watch
			return w[:-2]
		else: # if last one is consonant  macbooks -> macbook
			return w[:-1]
	return w


def find_noun(sent):
	'''	Returns a list of nouns if phrase nouns found
	'''
	noun_extractor = ConllExtractor()

	#nouns = noun_extractor.extract(sent)

	entities = []

	#if len(nouns) == 0:
	text = word_tokenize(sent)

	for w, pos in nltk.pos_tag(text):
		#print(w, pos)
		if len(w) > 1:
			if pos == 'NN' or w in BRANDS or w in PRODUCTS_TYPES:
				if w not in YES:
					entities.append(w)
			elif pos == 'NNS' and w not in YES:
				entities.append(convert_singular(w))
			elif w in PRODUCT:
				entities.append(w)
	return entities


'''
def find_adjective(sent):
	#Find the adjective in sentence 
	

	adj = None 
	for w, p in sent.pos_tags:
		if p == 'JJ':
			adj = w
			break
	return adj 
'''

def find_verb(sent):
	''' Returns the verbs as a list
	'''
	verbs = []
	text = word_tokenize(sent)

	for v, pos in nltk.pos_tag(text):
		if pos.startswith('VB'): #is a verb
			verbs.append(v)
		elif v == "n't" and pos == 'RB':
			verbs.append(v)
		elif v == 'like':
			verbs.append(v)

	return verbs

def find_pos(sent):
	'''Given a parsed input, find the pronoun, direct noun, adjective and verb to match their input.
		Only one sentence for now...	
	'''
	pronoun = None 
	noun = None 
	adjective = None
	verb = None

	
	#pronoun = find_pronoun(sentence)
	noun = find_noun(sent)


	#adjective = find_adjective(sentence)
	verb = find_verb(sent)

	return noun , verb

def display_products(data, with_price=False):
	'''
	Display the products from the search
	'''
	global df_last_display


	if data.empty:
		print_error_msg()

	else: 	

		if not with_price:

			df_last_display = data
			print("\nBot: We found the following products you might be interested:\n")
			for prod in data['Product Name'].values:
				print("Bot: " + prod)


		else:
			print("Bot: Here are the monthly price for the products: \n")
			for prod,price  in data[['Product Name', 'Subscription Plan']].values:
				print("Bot: The monthly price for " + prod + " is: " + str(price) + " .")





def is_in_list(list_1, list_2):
	'''	Check if at least one element of list1 is in list2
	'''
	for i in list_1:
		if i in list_2:
			return True
	return False


def find_objects(list_entities, intent, verbs):
	'''Filter out the the list of entities according to the intent 
		it returns a DataFrame with the products found
	'''

	prod_df = df	
	prod = pd.DataFrame()

	left_entities = []

	last = ''

	#Search by product name
	if intent == 1:
		product_names = find_names_in_products(list_entities)
		product_synonyms = find_synonyms(list_entities)
		product_brands = find_brands(list_entities)

		prod = df.loc[ df['Product Name'].str.contains('|'.join(map(re.escape, product_names)), case=False) ]
		#for pro in product_names:
		#	prod = prod_df.loc[prod_df['Product Name'].str.contains(pro, case=False, regex=False)]

		return prod

	##Seach by brands
	elif intent == 2:
		#print(list_entities)

		for entity in list_entities:
			#nouns = entity.split()
			for noun in entity.split():
				if noun in BRANDS:
					if noun != last:
						last = noun
						result = df.loc[df['Brand'].str.contains(noun, case = False)]
						check = result.isin(prod)
						if not check.all().Brand:
							prod = pd.concat([prod, result])
				else:
					left_entities.append(noun)


		if len(left_entities) > 0 and not prod.empty:

			if is_in_list(left_entities, PRODUCT):		
				#Check for object types 
				result = prod.loc[prod['Product Name'].str.contains('|'.join(map(re.escape, left_entities)), case=False)]
				if not result.empty:
					return result 

			if is_in_list(left_entities, PRODUCTS_TYPES):
				result = prod.loc[prod['Synonyms'].str.contains('|'.join(map(re.escape, left_entities)), case=False)]
				if not result.empty:
					return result 

	# Search for synonyms 
	elif intent == 3:

		for entity in list_entities:


			#entity = 'new apple phone'
			nouns = entity.split()
			for n in nouns:
				if n in PRODUCTS_TYPES:
					if n != last:
						last = n
						result = df.loc[df['Synonyms'].str.contains(n, case=False)]
						check = result.isin(prod)
						if not check.all().Brand:	
							prod = pd.concat([result, prod])
						
				else:
					left_entities.append(n)

		#Keep filtering for left entities
		if len(left_entities) > 0 and not prod.empty:
			if is_in_list(left_entities, BRANDS):
				#Check for brands 
				result = prod.loc[prod['Brand'].str.contains('|'.join(map(re.escape, left_entities)), case=False)]
				if not result.empty:
					return result 
			if is_in_list(left_entities, PRODUCT):
				result = prod.loc[prod['Product Name'].str.contains('|'.join(map(re.escape, left_entities)), case=False)]
				if not result.empty:
					return result 


	#Dislike 
	elif intent == 4:

		n_verb = len(verbs)
		n_entities = len(list_entities)

		

		if n_entities == 2:
			if "n't" in verbs[:2]: 
				#I dont like x , better y
				ent = list_entities[1]
				
				if df_last_display.empty:
					if ent in BRANDS:
						result = df.loc[df['Brand'].str.contains(ent , case=False, regex=False)]

					elif ent in PRODUCT:
						result = df.loc[df['Product Name'].str.contains(ent, case=False, regex =False)]
		
					elif ent in PRODUCTS_TYPES:
						result = df.loc[df['Synonyms'].str.contains(nouns, case=False, regex=False)]
					
					return result

				else:
					if ent in BRANDS:
						result = df_last_display.loc[df_last_display['Brand'].str.contains(ent , case=False, regex=False)]

					elif ent in PRODUCT:
						result = df_last_display.loc[df_last_display['Product Name'].str.contains(ent, case=False, regex=False)]
		
					elif ent in PRODUCTS_TYPES:
						result = df_last_display.loc[df_last_display['Synonyms'].str.contains(ent, case=False, regex=False)]
					
					return result					


		elif n_entities == 1: #Find something better

			ent = list_entities[0]
			#find products not same brand nor type nor name

			ent = convert_singular(ent)

			if df_last_display.empty:
				if ent in BRANDS:
					left_brands = BRANDS
					left_brands.remove(ent)
					#Found brand that user doesnt like
					result = df.loc[df['Brand'].str.contains('|'.join(map(re.escape, left_brands)), case=False)]
					return result

			else:
				left_brands = [ x.lower() for x in df_last_display['Brand'].values]
				left_brands = set(left_brands)

				left_names = []


				if ent in left_brands:
					left_brands.remove(ent)
					result = df_last_display.loc[df_last_display['Brand'].str.contains('|'.join(map(re.escape, left_brands)), case=False)]
					return result

				left_products = []
				for prod_name in df_last_display['Product Name'].values:

					if ent not in prod_name.lower():
						left_products.append(prod_name)

				if len(left_products) > 0:
					result = df_last_display.loc[df_last_display['Product Name'].str.contains('|'.join(map(re.escape, left_products)), case=False)]
					return result




	return prod



def do_intent(intent, entities, verbs):
	'''
	Execute the different intents:
	1: Search by product name
	2: Search by brand 
	3: Search by object tyope 
	4: Understand dislike and come up with something different
	'''
	products = find_objects(entities, intent, verbs)
	
	if products.empty:
		print_error_msg()

	else:
		display_products(products)
		question_interest(products)


def show_all(products):
	'''It simply displays the product price of the products data frame
	'''


	for name, price in products[['Product Name', 'Subscription Plan']].values:
			print("Bot: The montly price for " + name + " is: " + str(price) + " .")


def find_names_in_products(list_names):
	'''It searches for the names in list_names that match the product name
	'''
	found_names = []

	for name in list_names:
		if name in PRODUCT:
			found_names.append(name)

	return found_names

def find_brands(list_names):
	'''It searches brand names in list_names that match the product name
	'''
	found_brands = []

	for name in list_names:
		if name in BRANDS:
			found_brands.append(name)
	return found_brands



def find_synonyms(list_names):
	'''It searches product synonyms in list_names that match the product name
	'''
	found_syn = []

	for name in list_names:
		if name in PRODUCTS_TYPES:
			found_syn.append(name)
	return found_syn


def search_price_print(search_product, products_df):
	''' 
		Df is a dataframe with only products similar to the users choice
		search_product : input list with names of the product
		products_df : data frame with products queried from first search
	'''

	product_names = find_names_in_products(search_product)
	brand_names = find_brands(search_product)
	synonym_names = find_synonyms(search_product)
	found_prods = products_df


	#Check for brand/type/name  and for one or many words

	if len(product_names) > 0:
		for prod in product_names:
			found_prods = found_prods.loc[found_prods['Product Name'].str.contains(prod, case=False, regex=False)]

	if len(brand_names) > 0:
		for prod in brand_names:
			found_prods = found_prods.loc[found_prods['Brand'].str.contains(prod, case=False)]

	if len(synonym_names) > 0:
		for prod in synonym_names:
			found_prods = found_prods.loc[found_prods['Synonyms'].str.contains(prod, case=False)]


	if found_prods.empty:
		print_not_understand()
	else:
		show_all(found_prods)


def positive_answer(ans):
	'''Find a yes inside a answer typed by user
	'''
	ans = ans.lower()


	if 'y' in ans and len(ans) < 2:
		return True

	for y_word in YES:
		if y_word in ans:
			return True
	return False


def short_negative(ans): #Not, no, not at all
	'''Check if an answer is negative, it contains a not for answer
	'''
	NEG = ['no', 'not', 'nope', 'nop']

	if any(word in ans for word in NEG) and len(ans.split())==1:
		return True

	if ans == 'n':
		return 	True


	return False





def question_interest(products):
	''' Find out the interest product from a list of products 
	'''
	print()
	print("Bot: Are you interested in any of the products shown y/n?")
	ans = input("User: ")
	clean_ans = preprocess_text(ans)

	if positive_answer(ans):
		
		entities = find_noun(clean_ans)

		if len(products) == 1:
			#Only one product to show, just show it
			show_all(products)

		elif len(entities) > 0:	
			#its a positive answer and with name of product
			search_price_print(entities, products)
	
		else:
			print("Bot: In which of the products are you interested?")
			ans = input("User: ")
			clean_ans = preprocess_text(ans)
			entities = find_noun(clean_ans)


			if 'all' in clean_ans:
				show_all(products)

			elif short_negative(clean_ans) and len(entities) > 0:
				respond(clean_ans)

			else:
				search_price_print(entities, products)


	elif short_negative(clean_ans): #its only a no
		print("Bot: I am sorry to hear that, we will have more products very soon!")

	else:
		respond(clean_ans)

	


def find_intent(sentence):
	'''
	Funtion that maps a input sentence from the user to a intent 
	0: Search Price of products 
	1: Search similar products
	2: Search products by brand 
	'''

	#Create the features with the transformer
	features = transformer.transform([sentence])

	#If the prediction is not so sure from the user input return -1
	proba = classifier.predict_proba(features)[0]

	proba = np.array(proba)
	#print(proba)

	#if (proba > 0.5).sum() > 0:
	return classifier.predict(features)[0]

	#return -1


def train_classifier():
	'''It trains a classifier on handed made data. 
		It returns the classifier and the bag of words model 
	'''
	#Load data
	sentences, labels = create_data()

	#Create features - bag of words
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, stop_words = None, max_features = 200)
	vectorizer_tfidf = TfidfVectorizer()

	#train_features = vectorizer.fit_transform(sentences)
	train_features = vectorizer_tfidf.fit_transform(sentences)
	train_features = train_features.toarray()

	#Train the randomforest 
	forest = RandomForestClassifier(n_estimators = 100, random_state = 42)
	forest.fit(train_features, labels)

	#Some test sentences
	test_sentences = ['I dont like apple, i like samsung', 'I am interested in a s8', 'show me price for alexa echo', 'I look for shoes', 'apple iphone', 'show samsung' ]
	clean_test = [preprocess_text(x) for x in test_sentences]
	#test_features = vectorizer.transform(clean_test)
	test_features = vectorizer_tfidf.transform(clean_test)
	#print(forest.predict(test_features))

	return vectorizer_tfidf, forest



def respond(sentence):
	'''Parse the user's input sentence and find candidate terms the best-fit response'''
	cleaned = preprocess_text(sentence)



	#parsed = TextBlob(cleaned)
	#Loop through all the sentences, if more that one. To extract the most relevant
	#response text across multiple sentences
	# verb: list of verbs found
	# noun:  a list of nouns 
	nouns, verbs = find_pos(cleaned)
	intent = find_intent(cleaned)

	if len(nouns) == 0:
		print_not_understand()
	else:
		do_intent(intent, nouns, verbs)


def read_process(input_user):
	global df_last_display

	df_last_display = pd.DataFrame()
	respond(input_user)


	print("")
	print("Bot: Type for a new search :)")


	
if __name__ == '__main__':
	global transformer, classifier

	add_synonyms()
	transformer, classifier = train_classifier()

	print("Bot: Hello, I am your personal assistant, how can I help you?")
	print("Bot: Type what are you looking for?")

	try:
		while True:
			sentence = input("User: ")
			if 'exit' in sentence.lower():
				print('Bot: Have a nice day!')
				break
			read_process(sentence)

	except KeyboardInterrupt:
		print('Bot: Have a nice day!')

