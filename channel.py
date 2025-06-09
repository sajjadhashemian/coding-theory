import numpy as np
import math

class binary_symmetric_channel:
	def __init__(self, p):
		"""
		Initialize the Binary Symmetric Channel with a given error probability p.
		
		:param p: Probability of bit flip (0 <= p <= 1)
		"""
		if not (0 <= p <= 1):
			raise ValueError("Error probability p must be in the range [0, 1].")
		self.p = p

	def transmit(self, x):
		"""
		Transmit a binary sequence through the binary_symmetric_channel.
		
		:param x: Input binary sequence as a numpy array of 0s and 1s.
		:return: Output binary sequence after transmission.
		"""
		if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.integer):
			raise ValueError("Input must be a numpy array of integers (0s and 1s): {}".format(x))
		
		noise = np.random.binomial(1, self.p, size=x.shape)
		return np.bitwise_xor(x, noise)
	
	def capacity(self):
		"""
		Calculate the capacity of the Binary Symmetric Channel.
		
		:return: Capacity in bits per channel use.
		"""
		if self.p < 0 or self.p > 1:
			raise ValueError("Error probability p must be in the range [0, 1].")
		
		return 1 - self.p * np.log2(self.p) - (1 - self.p) * np.log2(1 - self.p) if self.p < 1 else 0

	def __str__(self):
		"""
		String representation of the binary_symmetric_channel.
		
		:return: String describing the binary_symmetric_channel with its error probability.
		"""
		return f"binary_symmetric_channel(p={self.p})"

class q_ary_symmetric_channel:
	def __init__(self, p, q):
		"""
		Initialize the q-ary symmetric channel with a given error probability p and alphabet size q.
		
		:param p: Probability of symbol flip (0 <= p <= 1)
		:param q: Size of the alphabet (q > 1)
		"""
		if not (0 <= p <= 1):
			raise ValueError("Error probability p must be in the range [0, 1].")
		if q <= 1:
			raise ValueError("Alphabet size q must be greater than 1.")
		
		self.p = p
		self.q = q

	def transmit(self, x):
		"""
		Transmit a q-ary sequence through the channel.
		
		:param x: Input sequence as a numpy array of integers in the range [0, q-1].
		:return: Output sequence after transmission.
		"""
		if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.integer):
			raise ValueError(f"Input must be a numpy array of integers: {x}")
		
		if np.any((x < 0) | (x >= self.q)):
			raise ValueError(f"Input values must be in the range [0, {self.q - 1}].")
		
		n = x.size
		output = np.copy(x)
		
		for i in range(n):
			if np.random.rand() < self.p:
				output[i] = np.random.randint(0, self.q)
		
		return output
	
	def capacity(self):
		"""
		Calculate the capacity of the q-ary symmetric channel.
		
		:return: Capacity in bits per channel use.
		"""
		if self.p < 0 or self.p > 1:
			raise ValueError("Error probability p must be in the range [0, 1].")
		
		return np.log2(self.q) * (1 - self.p)

	def __str__(self):
		return f"q-ary Symmetric Channel(p={self.p}, q={self.q})"
	
class channel:
	def __init__(self, p, q=None):
		"""
		Initialize a channel, either binary or q-ary symmetric channel.
		
		:param p: Error probability (0 <= p <= 1)
		:param q: Alphabet size (only for q-ary channel, must be > 1)
		"""
		if q is None:
			self.channel = binary_symmetric_channel(p)
		else:
			self.channel = q_ary_symmetric_channel(p, q)

	def transmit(self, x):
		return self.channel.transmit(x)

	def capacity(self):
		return self.channel.capacity()

	def __str__(self):
		return str(self.channel)