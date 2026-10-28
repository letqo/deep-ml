def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	result_vect = []
	if len(a[0]) != len(b):
		return -1
	else:
		for i, vect in enumerate(a):
			row_sum = 0
			for j in range(len(b)) :
				row_sum += b[j] * vect[j] 
			result_vect.append(row_sum)
			row_sum = 0
		return result_vect