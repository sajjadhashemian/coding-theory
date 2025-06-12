import numpy as np
from fractions import Fraction


class galois_field:
    """Simple representation of a Galois field.

    If p is None, operations are over the field of rational numbers
    using fractions.Fraction. Otherwise operations are modulo p for integers.
    """

    class Element:
        __slots__ = ("field", "value")

        def __init__(self, field, value):
            self.field = field
            # Normalize value into the field:
            if isinstance(value, galois_field.Element):
                # If already an Element: ensure same field
                if value.field is not field:
                    raise ValueError("Cannot create element: mismatched field.")
                self.value = value.value
            else:
                # Raw scalar: wrap into Fraction or int mod p
                if field.p is None:
                    # Rational field
                    self.value = Fraction(value)
                else:
                    val = int(value)
                    self.value = val % field.p

        def __repr__(self):
            return f"{self.value}"
            if self.field.p is None:
                return f"QElement({self.value})"
            else:
                return f"GFElement({self.field.p}, {self.value})"

        def __eq__(self, other):
            # Only compare to another Element or raw scalar
            if isinstance(other, galois_field.Element):
                return (self.field is other.field) and (self.value == other.value)
            else:
                # Try wrapping raw scalar
                try:
                    other_el = self.field(other)
                except Exception:
                    return False
                return self.value == other_el.value

        def __add__(self, other):
            # If other is an array of Elements or scalars, do elementwise addition:
            if isinstance(other, np.ndarray):
                # Vectorize: for each entry x in array, compute self + x
                vec = np.vectorize(lambda x: self + x, otypes=[object])
                return vec(other)
            # Otherwise wrap scalar and proceed
            other_el = self.field(other)
            if self.field.p is None:
                res = self.value + other_el.value
            else:
                res = (self.value + other_el.value) % self.field.p
            return galois_field.Element(self.field, res)

        def __radd__(self, other):
            return self.__add__(other)

        def __neg__(self):
            if self.field.p is None:
                res = -self.value
            else:
                res = (-self.value) % self.field.p
            return galois_field.Element(self.field, res)

        def __sub__(self, other):
            if isinstance(other, np.ndarray):
                vec = np.vectorize(lambda x: self - x, otypes=[object])
                return vec(other)
            other_el = self.field(other)
            if self.field.p is None:
                res = self.value - other_el.value
            else:
                res = (self.value - other_el.value) % self.field.p
            return galois_field.Element(self.field, res)

        def __rsub__(self, other):
            if isinstance(other, np.ndarray):
                vec = np.vectorize(lambda x: x - self, otypes=[object])
                return vec(other)
            other_el = self.field(other)
            return other_el.__sub__(self)

        def __mul__(self, other):
            if isinstance(other, np.ndarray):
                vec = np.vectorize(lambda x: self * x, otypes=[object])
                return vec(other)
            other_el = self.field(other)
            if self.field.p is None:
                res = self.value * other_el.value
            else:
                res = (self.value * other_el.value) % self.field.p
            return galois_field.Element(self.field, res)

        def __rmul__(self, other):
            if isinstance(other, np.ndarray):
                vec = np.vectorize(lambda x: x * self, otypes=[object])
                return vec(other)
            return self.__mul__(other)

        def inv(self):
            """Multiplicative inverse as an Element."""
            if self.field.p is None:
                # Rational: inverse of Fraction
                if self.value == 0:
                    raise ZeroDivisionError("Division by zero in rational field.")
                return galois_field.Element(self.field, Fraction(1, self.value))
            else:
                # Modular inverse in GF(p)
                if self.value % self.field.p == 0:
                    raise ZeroDivisionError("Division by zero in GF(p).")
                inv_val = pow(int(self.value), -1, self.field.p)
                return galois_field.Element(self.field, inv_val)

        def __truediv__(self, other):
            if isinstance(other, np.ndarray):
                # self / array: elementwise
                vec = np.vectorize(lambda x: self / x, otypes=[object])
                return vec(other)
            other_el = self.field(other)
            return self * other_el.inv()

        def __rtruediv__(self, other):
            if isinstance(other, np.ndarray):
                vec = np.vectorize(lambda x: x / self, otypes=[object])
                return vec(other)
            other_el = self.field(other)
            return other_el * self.inv()

        def __pow__(self, exponent):
            if not isinstance(exponent, int):
                raise TypeError("Exponent must be integer for field element power.")
            if self.field.p is None:
                # Fraction exponentiation
                # Note: could raise if denominator not integer power, but we assume integer exponent
                return galois_field.Element(self.field, self.value**exponent)
            else:
                # pow mod p
                val = pow(int(self.value), exponent, self.field.p)
                return galois_field.Element(self.field, val)

        def __int__(self):
            """
            Convert this field element to a Python int.
            - For GF(p), returns the integer in [0, p-1].
            - For the rational field, only allows conversion if the Fraction is integral.
            """
            if self.field.p is None:
                # Rational field: only allow if denominator == 1
                if isinstance(self.value, Fraction):
                    if self.value.denominator != 1:
                        raise ValueError(
                            f"Cannot convert non-integer Fraction {self.value} to int"
                        )
                    return self.value.numerator  # or int(self.value)
                else:
                    # If you stored something else, coerce to int
                    return int(self.value)
            else:
                # Finite field GF(p): stored value is already int mod p
                return int(self.value)

        def __float__(self):
            """
            Convert this field element to a float. Useful if you want a floating approximation
            (for rational field or GF(p) interpreted as integer).
            """
            # e.g. float(self.value); for GF(p), returns float of the integer representative.
            return float(self.value)

    def __init__(self, p=None):
        if p is not None:
            if p <= 1:
                raise ValueError("Modulus must be greater than 1 for a finite field")
            # Simple primality check
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    raise ValueError(
                        "Modulus must be a prime number for a finite field"
                    )
        self.p = p

    def __call__(self, value):
        """Wrap a scalar or array-like into Element(s)."""
        # If already an Element of this field
        if isinstance(value, galois_field.Element):
            if value.field is not self:
                raise ValueError("Cannot wrap Element from a different field.")
            return value

        # If value is a numpy array of dtype object whose elements are Elements of this field,
        # return as-is. Otherwise, if it's a numpy array of numeric scalars, wrap each entry.
        if isinstance(value, np.ndarray):
            # Force dtype object to hold Elements
            # We assume value is a numeric array (e.g. dtype int, float), or object array.
            arr = value
            # If object-dtype and all entries are already Elements of this field, return directly:
            if arr.dtype == object:
                all_el = True
                for x in arr.flat:
                    if not isinstance(x, galois_field.Element) or x.field is not self:
                        all_el = False
                        break
                if all_el:
                    return arr
            # Otherwise, wrap each entry into an Element
            vec = np.vectorize(lambda x: galois_field.Element(self, x), otypes=[object])
            return vec(arr)

        # If it is not a numpy array, but is array-like (e.g., list/tuple), convert via np.array with dtype object
        try:
            arr2 = np.array(value, dtype=object)
            # If this yields a 0-d array (i.e., a scalar), handle below:
            if arr2.shape == ():  # zero-dimensional array
                # treat as scalar
                return galois_field.Element(self, arr2.item())
            # Otherwise wrap each entry:
            vec = np.vectorize(lambda x: galois_field.Element(self, x), otypes=[object])
            return vec(arr2)
        except Exception:
            # Not array-like: scalar
            return galois_field.Element(self, value)


class vector_space:
    """Vector space over a given field, using pure Python lists for elimination."""

    def __init__(self, field: galois_field):
        self.field = field

    def row_echelon_form(self, matrix):
        """
        Compute row echelon form over the field.
        matrix can be a list of lists or a numpy array; we convert to list of lists of Elements.
        Returns a numpy array of dtype object containing Elements in echelon form.
        """
        # Build list-of-lists of Elements
        # If input is numpy array, convert to nested lists
        mat = np.asarray(matrix)
        # Convert to Python nested lists for clarity
        A = [
            [self.field(mat[i, j]) for j in range(mat.shape[1])]
            for i in range(mat.shape[0])
        ]
        m = len(A)
        n = len(A[0]) if m > 0 else 0
        r = 0
        zero = self.field(0)
        for c in range(n):
            # Find pivot in column c at or below row r
            pivot = None
            for i in range(r, m):
                if A[i][c] != zero:
                    pivot = i
                    break
            if pivot is None:
                # Entire column zero in rows r..m-1
                continue
            # Swap pivot row into position r
            if pivot != r:
                A[r], A[pivot] = A[pivot], A[r]
            pivot_val = A[r][c]
            if pivot_val == zero:
                # This should not happen, since we found A[r][c] != zero
                raise ZeroDivisionError(f"Unexpected zero pivot at row {r}, column {c}")
            # Normalize pivot row: divide entire row by pivot_val
            inv_p = pivot_val.inv()
            for j in range(c, n):
                A[r][j] = A[r][j] * inv_p
            # Eliminate entries below pivot
            for i in range(r + 1, m):
                factor = A[i][c]
                if factor != zero:
                    # row_i := row_i - factor * row_r
                    for j in range(c, n):
                        A[i][j] = A[i][j] - factor * A[r][j]
            r += 1
            if r == m:
                break
        # Convert back to numpy array of dtype object
        return np.array(A, dtype=object)

    def reduced_row_echelon_form(self, matrix):
        """
        First compute row echelon, then eliminate above pivots.
        """
        # Get row echelon as numpy array, convert to nested lists
        echelon = self.row_echelon_form(matrix)
        A = echelon.tolist()
        m = len(A)
        n = len(A[0]) if m > 0 else 0
        zero = self.field(0)
        # Work from bottom row up
        for i in range(m - 1, -1, -1):
            # Find pivot column in row i
            pivot_col = None
            for j in range(n):
                if A[i][j] != zero:
                    pivot_col = j
                    break
            if pivot_col is None:
                continue
            # Row is assumed normalized so that A[i][pivot_col] == 1
            # Eliminate above
            for k in range(i):
                factor = A[k][pivot_col]
                if factor != zero:
                    for j in range(pivot_col, n):
                        A[k][j] = A[k][j] - factor * A[i][j]
        return np.array(A, dtype=object)


if __name__ == "__main__":
    # Quick tests

    # 1) Finite field GF(7)
    gf = galois_field(7)
    vs = vector_space(gf)

    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
    # Over GF(7), row3 becomes [0,1,2] after mod 7, so echelon form is valid.
    ref = vs.row_echelon_form(matrix)
    rref = vs.reduced_row_echelon_form(matrix)
    print("GF(7) Row Echelon Form:")
    print(ref)
    print("GF(7) Reduced Row Echelon Form:")
    print(rref)

    # 2) Rational field
    gfq = galois_field(None)
    vq = vector_space(gfq)
    mat_rat = [[1, 2, 3], [2, 4, 5], [3, 6, 9]]
    rref_rat = vq.reduced_row_echelon_form(mat_rat)
    print("Rational Reduced Row Echelon Form:")
    print(rref_rat)

    # 3) Simple vector addition / scalar multiplication tests
    a = [1, 2, 3]
    b = [4, 5, 6]
    a_gf = gf(a)
    b_gf = gf(b)

    # Direct element arithmetic
    x = gf(2)
    y = gf(5)
    print("x + y =", x + y)
    print("x * y =", x * y)
    print("x / y =", x / y)
    print("Inverse of x:", x.inv())
    print("Negation of x:", -x)

    # Rational examples
    a_rat = gfq([1, 2, 3])
    b_rat = gfq([Fraction(4, 3), Fraction(5, 2), 6])
    print("Rational addition:", a_rat + b_rat)
    print("Rational scalar mul:", gfq(Fraction(3, 2)) * a_rat)

    # Example usage of Galois field and vector space with operators

    # 1) Finite field GF(7)
    gf = galois_field(7)
    vs = vector_space(gf)

    # Create field elements / vectors:
    # Method A: wrap Python scalars / lists directly:
    a = gf([1, 2, 3])  # yields array of Elements
    b = gf([4, 5, 6])

    # Direct operator:
    print("Addition via operator +:", a + b)
    print("Subtraction via operator -:", a - b)
    print("Scalar multiplication via operator *:", gf(3) * a)

    # Row echelon and RREF:
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ref = vs.row_echelon_form(matrix)
    rref = vs.reduced_row_echelon_form(matrix)
    print("Row Echelon Form:\n", ref)
    print("Reduced Row Echelon Form:\n", rref)

    # Direct field element operations:
    x = gf(2)
    y = gf(5)
    print("x, y:", x, y)
    print("x + y =", x + y)
    print("x - y =", x - y)
    print("x * y =", x * y)
    print("x / y =", x / y)
    print("Inverse of x:", x.inv())
    print("Negation of x:", -x)

    # 2) Rational field (p=None)
    gfq = galois_field(None)
    vq = vector_space(gfq)
    # wrap scalars or arrays:
    a_rat = gfq([1, 2, 3])
    b_rat = gfq([Fraction(4, 3), Fraction(5, 2), 6])
    print("Rational field addition:", a_rat + b_rat)
    print("Rational scalar mul:", gfq(Fraction(3, 2)) * a_rat)

    # Row echelon on rationals:
    mat_rat = [[1, 2, 3], [2, 4, 5], [3, 6, 9]]
    print("Rational RREF:\n", vq.reduced_row_echelon_form(mat_rat))
