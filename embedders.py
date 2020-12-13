from glove_embed import GloVe
from bert_embed import BERT
from use_embed import USE
import pandas as pd
import numpy as np
import os
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

class WordLookUpSimilarty:

	def __init__(self):
		self.path_to_tags = './product_wise_tags.csv'
		self.save_dir = 'combined_tags_glove'
		os.system('mkdir '+self.save_dir)
		self.confidence = 0.85
		self.depth_modifier = 0.825
		self.glove = GloVe()
		self.similarity_importance = 0.5
		self.depth_similarity = True
		self.disable = False
		self.average_tags_removed = []

	def similarity(self, a, b):
		if norm(a) == 0 or norm(b) == 0:
			cos_sim = 0
		else:
			cos_sim = dot(a, b)/(norm(a)*norm(b))
		return cos_sim

	def all_tags_similarity(self, tag_embeds):
		similarity = np.zeros((len(tag_embeds), len(tag_embeds)))
		for i in range(len(tag_embeds)):
			for j in range(len(tag_embeds)):
				similarity_score = self.similarity(tag_embeds[i], tag_embeds[j])
				similarity[i][j] =  similarity_score
		return similarity

	def extract(self):
		tags_ensemble = pd.read_csv(self.path_to_tags)
		tags_ensemble = tags_ensemble.values
		products = tags_ensemble.T[0]
		tags_ensemble = tags_ensemble.T[1:].T
		for product, tags in tqdm(zip(products, tags_ensemble), total=len(products), disable=self.disable, desc='Clustering Tags-per-Product using WordLookUpSimilarty Methodology'):
			tag_embeds = np.zeros((len(tags), 100))
			for i, tag in enumerate(tags):
				embeds = np.mean(self.glove.glove_extract(tag), axis=0)
				tag_embeds[i] = embeds
			similarity = self.all_tags_similarity(tag_embeds)
			new_similarity = np.zeros(similarity.shape)
			for row in range(similarity.shape[0]):
				for col in range(similarity.shape[1]):
					confidence_item = similarity[row][col]
					new_similarity[row] += (confidence_item * similarity[col])/12
			
			if self.depth_similarity:
				similarity = new_similarity
				self.confidence = self.confidence * self.depth_modifier

			positions = (12 - np.arange(12))/120
			importance = self.similarity_importance*np.mean(similarity, axis=0)+(1-self.similarity_importance)*positions
			importance_order = np.argsort(importance)[::-1]
			words_already_clustered = []
			clusters = []
			for index in importance_order:
				cluster = []
				for row in importance_order:
					if 1 >= similarity[row][index] > self.confidence and tags[row] not in words_already_clustered:
						words_already_clustered.append(tags[row])
						cluster.append(tags[row])
				if len(cluster)>1:
					clusters.append(cluster)
			row = pd.DataFrame({0:['cluster-head'], 1:['similar-tags']})
			row.to_csv('./'+self.save_dir+'./'+product+'.csv', mode='w', header=False, index=False)
			for cluster in clusters:
				row = pd.DataFrame({i:[c] for i,c in enumerate(cluster)})
				row.to_csv('./'+self.save_dir+'./'+product+'.csv', mode='a', header=False, index=False)
			self.average_tags_removed.append(len(words_already_clustered))

class ContextualWordEmbeddingsSimilarity:

	def __init__(self):
		self.path_to_tags = './product_wise_tags.csv'
		self.save_dir = 'combined_tags_bert'
		os.system('mkdir '+self.save_dir)
		self.confidence = 0.85
		self.depth_modifier = 0.825
		self.similarity_importance = 0.5
		self.depth_similarity = False
		self.disable = False
		self.average_tags_removed = []
		self.bert = BERT()

	def similarity(self, a, b):
		if norm(a) == 0 or norm(b) == 0:
			cos_sim = 0
		else:
			cos_sim = dot(a, b)/(norm(a)*norm(b))
		return cos_sim

	def all_tags_similarity(self, tag_embeds):
		similarity = np.zeros((len(tag_embeds), len(tag_embeds)))
		for i in range(len(tag_embeds)):
			for j in range(len(tag_embeds)):
				similarity_score = self.similarity(tag_embeds[i], tag_embeds[j])
				similarity[i][j] =  similarity_score
		return similarity

	def extract(self):
		tags_ensemble = pd.read_csv(self.path_to_tags)
		tags_ensemble = tags_ensemble.values
		products = tags_ensemble.T[0]
		tags_ensemble = tags_ensemble.T[1:].T
		for product, tags in tqdm(zip(products, tags_ensemble), total=len(products), disable=self.disable, desc='Clustering Tags-per-Product using ContextualWordEmbeddingsSimilarity Methodology'):
			tag_embeds = self.bert.bert_extract(tags)
			tag_embeds = np.array([np.mean(i, axis=0) for i in tag_embeds])
			similarity = self.all_tags_similarity(tag_embeds)
			new_similarity = np.zeros(similarity.shape)
			for row in range(similarity.shape[0]):
				for col in range(similarity.shape[1]):
					confidence_item = similarity[row][col]
					new_similarity[row] += (confidence_item * similarity[col])/12
			
			if self.depth_similarity:
				similarity = new_similarity
				self.confidence = self.confidence * self.depth_modifier

			positions = (12 - np.arange(12))/120
			importance = self.similarity_importance*np.mean(similarity, axis=0)+(1-self.similarity_importance)*positions
			importance_order = np.argsort(importance)[::-1]
			words_already_clustered = []
			clusters = []
			for index in importance_order:
				cluster = []
				for row in importance_order:
					if 1 >= similarity[row][index] > self.confidence and tags[row] not in words_already_clustered:
						words_already_clustered.append(tags[row])
						cluster.append(tags[row])
				if len(cluster)>1:
					clusters.append(cluster)
			row = pd.DataFrame({0:['cluster-head'], 1:['similar-tags']})
			row.to_csv('./'+self.save_dir+'./'+product+'.csv', mode='w', header=False, index=False)
			for cluster in clusters:
				row = pd.DataFrame({i:[c] for i,c in enumerate(cluster)})
				row.to_csv('./'+self.save_dir+'./'+product+'.csv', mode='a', header=False, index=False)
			self.average_tags_removed.append(len(words_already_clustered))

class SentenceEmbeddingsSimilarity:

	def __init__(self):
		self.path_to_tags = './product_wise_tags.csv'
		self.save_dir = 'combined_tags_use'
		os.system('mkdir '+self.save_dir)
		self.confidence = 0.75
		self.depth_modifier = 0.825
		self.similarity_importance = 0.5
		self.depth_similarity = False
		self.disable = False
		self.average_tags_removed = []
		self.use = USE()

	def similarity(self, a, b):
		if norm(a) == 0 or norm(b) == 0:
			cos_sim = 0
		else:
			cos_sim = dot(a, b)/(norm(a)*norm(b))
		return cos_sim

	def all_tags_similarity(self, tag_embeds):
		similarity = np.zeros((len(tag_embeds), len(tag_embeds)))
		for i in range(len(tag_embeds)):
			for j in range(len(tag_embeds)):
				similarity_score = self.similarity(tag_embeds[i], tag_embeds[j])
				similarity[i][j] =  similarity_score
		return similarity

	def extract(self):
		tags_ensemble = pd.read_csv(self.path_to_tags)
		tags_ensemble = tags_ensemble.values
		products = tags_ensemble.T[0]
		tags_ensemble = tags_ensemble.T[1:].T
		for product, tags in tqdm(zip(products, tags_ensemble), total=len(products), disable=self.disable, desc='Clustering Tags-per-Product using SentenceEmbeddingsSimilarity Methodology'):
			tag_embeds = self.use.use_extract(tags)
			similarity = self.all_tags_similarity(tag_embeds)
			new_similarity = np.zeros(similarity.shape)
			for row in range(similarity.shape[0]):
				for col in range(similarity.shape[1]):
					confidence_item = similarity[row][col]
					new_similarity[row] += (confidence_item * similarity[col])/12
			
			if self.depth_similarity:
				similarity = new_similarity
				self.confidence = self.confidence * self.depth_modifier

			positions = (12 - np.arange(12))/120
			importance = self.similarity_importance*np.mean(similarity, axis=0)+(1-self.similarity_importance)*positions
			importance_order = np.argsort(importance)[::-1]
			words_already_clustered = []
			clusters = []
			for index in importance_order:
				cluster = []
				for row in importance_order:
					if 1 >= similarity[row][index] > self.confidence and tags[row] not in words_already_clustered:
						words_already_clustered.append(tags[row])
						cluster.append(tags[row])
				if len(cluster)>1:
					clusters.append(cluster)
			row = pd.DataFrame({0:['cluster-head'], 1:['similar-tags']})
			row.to_csv('./'+self.save_dir+'./'+product+'.csv', mode='w', header=False, index=False)
			for cluster in clusters:
				row = pd.DataFrame({i:[c] for i,c in enumerate(cluster)})
				row.to_csv('./'+self.save_dir+'./'+product+'.csv', mode='a', header=False, index=False)
			self.average_tags_removed.append(len(words_already_clustered))

'''
wlus = WordLookUpSimilarty()
wlus.extract()

cwes = ContextualWordEmbeddingsSimilarity()
cwes.extract()
'''
ses = SentenceEmbeddingsSimilarity()
ses.extract()