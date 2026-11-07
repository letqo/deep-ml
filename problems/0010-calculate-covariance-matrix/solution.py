def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

	nb_features = len(vectors)
	N = len(vectors[0])
	mean_features = []
	row = []
	cov = []
	cov_ij = []
	
	for i in range(len(vectors)):
		mean_features.append(sum(vectors[i])/N)
	for i in range(nb_features):
		for j in range(nb_features):
			for k, l in zip(vectors[i], vectors[j]):
				cov_ij.append((k - mean_features[i]) * (l - mean_features[j]))
			row.append(sum(cov_ij)/(N - 1))
			cov_ij = []
		cov.append(row)
		row = []
	return cov