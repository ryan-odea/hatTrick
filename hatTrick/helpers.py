import numpy as np


def isprime(n: int) -> bool:
    if n < 2:
        return False
    for x in range(2, int(n**0.5) + 1):
        if n % x == 0:
            return False
    return True


def ishift(alist: list, n: int) -> list:
    for i in range(n):
        tmp = alist.pop(0)
        alist.append(tmp)
    return alist


def generate_s_matrix(n: int) -> np.ndarray:
    if not isprime(n):
        raise ValueError(f"n={n} must be prime for quadratic residues method")

    if n % 4 != 3:
        raise ValueError(f"n={n} must satisfy n ≡ 3 (mod 4) for this construction")

    m = range(0, n)
    Srow = [0 for _ in m]

    for i in range(0, (n - 1) // 2):
        Srow[(i + 1) * (i + 1) % n] = 1

    Srow[0] = 1

    S = [[0 for _ in m] for __ in m]

    rowcopy = Srow.copy()
    for i in m:
        for j in m:
            S[i][j] = rowcopy[j]
        rowcopy = ishift(rowcopy, 1)

    return np.array(S)


def compute_hadamard_inverse(S: np.ndarray) -> np.ndarray:
    n = S.shape[0]

    JN = np.ones((n, n))
    ST = S.T
    Sinv = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Sinv[i][j] = 2.0 * (2.0 * ST[i][j] - JN[i][j]) / (n + 1)

    return Sinv
