
from collections import Counter

def confusion_matrix(data):
	
	tuple_data = []
	for ls in data:
		tuple_data.append(tuple(ls))
	count = Counter(tuple_data)
	return [[count[(1,1)], count[(1,0)]], [count[(0,1)], count[(0,0)]]]