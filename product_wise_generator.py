import pandas as pd
import numpy as np
import os
from tqdm import tqdm

data = pd.read_csv('reviews_Cell_Phones_and_Accessories_5.csv')
f = data.columns
data = data.values
product_id = np.unique(data.T[1])
os.system('mkdir product_wise_reviews')

for p_id in tqdm(product_id):
	p = []
	for row in data:
		if row[1] == p_id:
			p.append(row)
	p = pd.DataFrame(np.array(p))
	p.columns = f
	p.to_csv('./product_wise_reviews/'+str(p_id)+'.csv', index=False)