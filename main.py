import numpy as np
import math
from channel import channel


if __name__=='__main__':
	p = 0.1  # Error probability

	# Create a binary symmetric channel
	binary_channel = channel(p)
	print(binary_channel)
	# Transmit a binary sequence
	binary_sequence = np.array([0, 1, 0, 1, 1, 0], dtype=int)
	output_binary = binary_channel.transmit(binary_sequence)
	print("Input binary sequence:", binary_sequence)
	print("Output binary sequence:", output_binary)
	print("Binary channel capacity:", binary_channel.capacity())


	p = 0.4  # Error probability
	q = 4    # Alphabet size for q-ary channel
	# Create a q-ary symmetric channel
	q_ary_channel = channel(p, q)
	print(q_ary_channel)
	# Transmit a q-ary sequence
	q_ary_sequence = np.array([0, 1, 2, 3, 0, 1], dtype=int)
	output_q_ary = q_ary_channel.transmit(q_ary_sequence)
	print("Input q-ary sequence:", q_ary_sequence)
	print("Output q-ary sequence:", output_q_ary)
	print("Q-ary channel capacity:", q_ary_channel.capacity())
	
