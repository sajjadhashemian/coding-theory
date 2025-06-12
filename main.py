import numpy as np
import math
from channel import channel
from common import galois_field, vector_space

if __name__ == "__main__":
    p = 0.3  # Error probability

    # Create a binary symmetric channel
    binary_channel = channel(p)
    print(binary_channel)
    # Transmit a binary sequence
    binary_sequence = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    output_binary = binary_channel.transmit(binary_sequence)
    print("Input binary sequence:", binary_sequence)
    print("Output binary sequence:", output_binary)
    print("Binary channel capacity:", binary_channel.capacity())

    # 1) Finite field GF(2)
    gf = galois_field(2)
    vs = vector_space(gf)

    matrix = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=int)
    rref = vs.reduced_row_echelon_form(matrix)
    print("GF(7) Reduced Row Echelon Form:")
    print(rref)
    output_binary = binary_channel.transmit(rref)
    print("Output binary sequence:")
    print(output_binary)
