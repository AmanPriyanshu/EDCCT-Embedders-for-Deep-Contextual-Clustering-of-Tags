import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_bert import extract_embeddings
import numpy as np

class BERT:
	def __init__(self):
		self.model_path = './bert/uncased_L-4_H-512_A-8'

	def bert_extract(self, texts):
		embeddings = extract_embeddings(self.model_path, texts)
		return np.array(embeddings)