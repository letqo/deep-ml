import math

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	s = []
	J = []
	J_row = []
	for z in x:
		s.append(math.exp(z - max(x)))
	s = [i/sum(s) for i in s] 
	
	for i in range(len(s)):
		for j in range(len(s)):
			entry = - s[i] * s[j] if i != j else s[i] * (1 - s[i])
			J_row.append(round(entry, 4))
		J.append(J_row)
		J_row = []
	return J
