class qubit():
	def __init__(self, zeroweight, oneweight, zeroket = None, oneket = None):
		self.zeroket = zeroket
		self.zeroweight = zeroweight
		self.oneket = oneket
		self.oneweight = oneweight

	def to_vector(self):
		return self.__to_vector(self, [], 1)

	def __to_vector(self, qubit, data, weight):
		if (qubit.zeroket is None):
			data.append(weight*qubit.zeroweight)
		else:
			data = qubit.__to_vector(qubit.zeroket, data, weight*qubit.zeroweight)
		if (qubit.oneket is None):
			data.append(weight*qubit.oneweight)
		else:
			data = qubit.__to_vector(qubit.oneket, data, weight*qubit.oneweight)
		return data
	
	def kron(self, target):
		return qubit(self.zeroweight, self.oneweight, target, target)