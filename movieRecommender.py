import numpy as np 
from lightfm.datasets import fetch_movielens
from lightfm import LightFM 

#fetch data of ratings higher than specified and format 
data = fetch_movielens(min_rating=4.0)

print(repr(data['train'])) #train Dataset
print(repr(data['test']))  #test dataset

#store model by using LightFM init the model
model = LightFM(loss='warp')

#train - epochs is number of runs, num_threads is parallel computations
model.fit(data['train'], epochs=30, num_threads=2)

#need to generate a recommendation using the model
def sample_recommendation(model, data, user_ids):

	#number of users and movies in training data
	n_users, n_items = data['train'].shape

	#generate recommendations for each user we input
	for user_id in user_ids:

		#if rating>5: +ve : if rating<5 : -ve
		#movies the user already likes
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#generate recommendations and store them in a scores variable
		#movies that our model predicts that user will like
		scores = model.predict(user_id, np.arange(n_items))

		#rank them in order of most liked to least 
		top_items = data['item_labels'][np.argsort(-scores)]

		#print out the results
		print("User %s" % user_id) #%s converts ID to string
		print("     Known positives:")

		for i in known_positives[:3]:
			print("            %s" % i)

		print("     Recommend:")

		for i in top_items[:3]:
			print("            %s" % i)

sample_recommendation(model, data, [3,25,450])