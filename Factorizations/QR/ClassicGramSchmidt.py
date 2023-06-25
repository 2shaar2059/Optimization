import numpy as np


def ClassicGramSchmidt(A):
    rows, cols = np.shape(A)
    Q = np.zeros_like(A)
    R = np.zeros((cols, cols))

    for i in range(cols):
        q = A[:, i].copy()
        j = 0
        while j < i:
            q_j = Q[:, j].copy()
            R[j, i] = np.dot(q, q_j)
            q -= q_j * R[j, i]
            j += 1
        R[j, i] = np.sqrt(np.dot(q, q))
        Q[:, i] = q / R[j, i]

    return Q, R


if __name__ == "__main__":
    A = np.array(
        [
            [1, 2, 3],
            [1, 3, 3],
            [2, 3, 4],
        ]
    ).astype(np.float64)
    print("A:\n", A)

    Q, R = ClassicGramSchmidt(A)

    print("Q:\n", Q)
    print("R:\n", R)
    print("Q.T * Q:\n", Q.T @ Q)
    print("Q * Q.T:\n", Q @ Q.T)
    print("QR:\n", Q @ R)
