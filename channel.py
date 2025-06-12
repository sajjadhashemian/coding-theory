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
        self.q = 2  # Binary channel has an alphabet size of 2 (0 and 1)

    def transmit(self, x, return_noise=False):
        """
        Transmit a binary sequence through the binary_symmetric_channel.

        :param x: Input binary sequence as a numpy array of 0s and 1s.
        :return: Output binary sequence after transmission.
        """
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.integer):
            raise ValueError(
                "Input must be a numpy array of integers (0s and 1s): {}".format(x)
            )

        noise = np.random.binomial(1, self.p, size=x.shape)
        if return_noise:
            return np.bitwise_xor(x, noise), noise
        else:
            return np.bitwise_xor(x, noise)

    def capacity(self):
        """
        Calculate the capacity of the Binary Symmetric Channel.

        :return: Capacity in bits per channel use.
        """
        if self.p < 0 or self.p > 1:
            raise ValueError("Error probability p must be in the range [0, 1].")

        return (
            1 - self.p * np.log2(self.p) - (1 - self.p) * np.log2(1 - self.p)
            if self.p < 1
            else 0
        )

    def __str__(self):
        """
        String representation of the binary_symmetric_channel.

        :return: String describing the binary_symmetric_channel with its error probability.
        """
        return f"Binary Symmetric Channel with p={self.p}."


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

    def transmit(self, x, return_noise=False):
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
        noise = np.zeros(n, dtype=int)

        for i in range(n):
            if np.random.rand() < self.p:
                output[i] = np.random.randint(0, self.q)
                noise[i] = 1

        if return_noise:
            return output, noise
        else:
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
        return f"q-ary Symmetric Channel with p={self.p}, q={self.q}."


class channel:
    def __init__(self, p, q=None, random_state=None):
        """
        Initialize a channel, either binary or q-ary symmetric channel.

        :param p: Error probability (0 <= p <= 1)
        :param q: Alphabet size (only for q-ary channel, must be > 1)
        :param random_state: Optional random state for reproducibility.
        """
        if random_state is not None:
            np.random.seed(random_state)
        if q is None:
            self.channel = binary_symmetric_channel(p)
        else:
            self.channel = q_ary_symmetric_channel(p, q)

    def transmit(self, x):
        x = np.asarray(x, dtype=int)
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f"Input must be a numpy array of integers: {x}")
        if np.any((x < 0) | (x >= self.channel.q)):
            raise ValueError(
                f"Input values must be in the range [0, {self.channel.q - 1}] for q-ary channel or [0, 1] for binary channel."
            )
        ret = []
        if len(x.shape) > 1:
            for i in range(x.shape[0]):
                y = self.channel.transmit(x[i])
                ret.append(y)
            return np.array(ret)
        return self.channel.transmit(x)

    def capacity(self):
        return self.channel.capacity()

    def __str__(self):
        return str(self.channel)


if __name__ == "__main__":
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
    q = 4  # Alphabet size for q-ary channel
    # Create a q-ary symmetric channel
    q_ary_channel = channel(p, q)
    print(q_ary_channel)
    # Transmit a q-ary sequence
    q_ary_sequence = np.array([0, 1, 2, 3, 0, 1], dtype=int)
    output_q_ary = q_ary_channel.transmit(q_ary_sequence)
    print("Input q-ary sequence:", q_ary_sequence)
    print("Output q-ary sequence:", output_q_ary)
    print("Q-ary channel capacity:", q_ary_channel.capacity())
