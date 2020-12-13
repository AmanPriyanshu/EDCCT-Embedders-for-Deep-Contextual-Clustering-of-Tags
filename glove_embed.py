import numpy as np
from tqdm import tqdm

class GloVe:
	def __init__(self):
		self.model_path = './glove/glove.6B.100d.txt'
		self.embeddings_index = None
		self.initialize_glove()

	def initialize_glove(self):
		self.embeddings_index = {}
		with open(self.model_path, encoding="utf8") as f:
			for line in tqdm(f, total=400000, desc='Extracting GloVe Embeddings'):
				values = line.split();
				word = values[0];
				coefs = np.asarray(values[1:], dtype='float32');
				self.embeddings_index[word] = coefs;

	def glove_extract(self, tag):
		embeddings = []
		for w in tag.split():
			try:
				embeddings.append(self.embeddings_index[w])
			except:
				pass
		if len(embeddings) == 0:
			embeddings = np.zeros((1, 100))
		return np.array(embeddings)