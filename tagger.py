import numpy as np
import os
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string
from nltk import ngrams
from collections import Counter

setup = False
if setup:
	nltk.download('stopwords')
	nltk.download('averaged_perceptron_tagger')

class TagGenerator:
	def __init__(self):
		self.stop = [i.lower() for i in stop]
		self.unimportant_words = stop + [i.lower() for i in "34 almost a about all and are as at back be because been but can can't come could did didn't do don't for from get go going good got had have he her here he's hey him his how I if I'll I'm in is it it's just know like look me mean my no not now of oh OK okay on one or out really right say see she so some something tell that that's the then there they think this time to up want was we well were what when who why will with would yeah yes you your you're".replace("'", "").split()]
		self.rejected_tags = ['PRP', 'NNP', 'WDT', 'WP', 'WP$', 'WRB', 'UH', 'PRP$', 'CC', 'IN', 'TO', 'MD', 'DT']
		self.w_dual = [0.3, 0.5]
		self.w_tri = [0.2, 0.3, 0.3]
		self.ids_weight = 0
		self.dual_weight, self.tri_weight = 0.5, 0.4
		self.disable = False

	def preprocess_sent(self, sent):
		sent = str(sent).replace("'", "")
		sent = nltk.word_tokenize(sent)
		sent = nltk.pos_tag(sent)
		sent = " ".join([i[0] for i in sent if i[1] not in self.rejected_tags])
		sent = "".join([w for w in sent if w not in string.punctuation])
		sent = " ".join([i for i in sent.split() if i not in self.unimportant_words])
		sent = sent.lower()
		return sent

	def preprocess(self, arr):
		arr = [self.preprocess_sent(i) for i in arr]
		return arr

	def n_grammer(self, arr, n):
		n = [n]
		vals = [y for x in arr for y in x.split()]
		rslt = [' '.join(y) for x in n for y in ngrams(vals, x)]
		grams = np.array(list(Counter(rslt).keys()))
		weights = np.array([i for _, i in Counter(rslt).items()])
		index = np.argsort(weights)[::-1]
		weights = weights[index]
		grams = grams[index]
		ids = np.log((len(arr)+1)/np.array([1 + sum([1 for sent in arr if w in sent]) for w in grams]))
		weights = weights * ids * self.ids_weight + weights * (1 - self.ids_weight)
		return grams, weights

	def tag_collection(self, arr):
		
		grams, weights = {i:None for i in range(1, 4)}, {i:None for i in range(1, 4)}
		for n in range(1, 4):
			grams[n], weights[n] = self.n_grammer(arr, n)

		tags_final = [t for t in grams[2][:6]]
		for t, w in zip(grams[3], weights[3]):
			if w>weights[2][0]/2:
				tags_final.append(t)
		for t, w in zip(grams[1], weights[1]):
			if w>weights[2][0]*4:
				tags_final.append(t)
		if len(tags_final) <= 12:
			tags_final = tags_final + [t for t in grams[3][:4]] + [t for t in grams[1][:4]]
		return tags_final

	def collect(self, dir_path='./product_wise_reviews/', save_path='./product_wise_tags.csv', return_tags=False, start=0, end=None):
		if return_tags:
			collected_tags = []
		dir_paths = sorted([dir_path+i for i in os.listdir(dir_path) if '.csv' in i])
		header = True
		mode = 'w'
		if end==None:
			end = len(dir_paths)
		for path in tqdm(dir_paths[start:end], disable=self.disable):
			df = pd.read_csv(path, usecols=['reviewText'])
			df = df.values
			df = df.flatten()
			df = self.preprocess(df)
			tags = self.tag_collection(df)[:12]
			if return_tags:
				collected_tags.append(tags)
			else:
				tags_output = {'product_id':[path[len(dir_path):-4]]}
				tags_output.update({'tag_'+str(i+1):[tags[i]] for i in range(12)})
				tags_output = pd.DataFrame(tags_output)
				tags_output.to_csv(save_path, header=header, index=False, mode=mode)
				header = False
				mode='a'
		if return_tags:
			return collected_tags

def main():
	tg = TagGenerator()
	tags = tg.collect()

main()