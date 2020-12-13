import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_bert import extract_embeddings
import tensorflow_hub as hub
import numpy as np

class USE:
	def __init__(self):
		self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
		self.embed = hub.load(self.module_url)

	def use_extract(self, texts):
		embeddings = self.embed(texts)
		return np.array(embeddings)