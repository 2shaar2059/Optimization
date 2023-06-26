import numpy as np


def ModifiedGramSchmidt_old(A):
    rows, cols = np.shape(A)
    Q = np.zeros_like(A)
    R = np.zeros((cols, cols))

    for i in range(cols):
        q_i = A[:, i].copy()
        j = 0
        while j < i:
            q_j = Q[:, j].copy()
            R[j, i] = np.dot(q_i, q_j)
            q_i -= q_j * R[j, i]
            j += 1
        R[j, i] = np.sqrt(np.dot(q_i, q_i))
        Q[:, i] = q_i / R[j, i]  # TODO handle divide-by-0
    return Q, R


def ModifiedGramSchmidt(A):
    rows, cols = np.shape(A)
    Q = A.copy()
    R = np.zeros((cols, cols))
    for i in range(cols):
        q_i = Q[:, i]
        R[i, i] = np.sqrt(np.dot(q_i, q_i))
        q_i /= R[i, i]  # TODO handle divide-by-0
        for j in range(i + 1, cols):
            q_j = Q[:, j]
            R[i, j] = np.dot(q_j, q_i)
            q_j -= q_i * R[i, j]
    return Q, R


if __name__ == "__main__":
    A = np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
            [1, 1, 0],
        ]
    ).astype(np.float64)
    print("A:\n", A)

    Q1, R1 = ModifiedGramSchmidt_old(A)
    print("Q1:\n", Q1)
    print("R1:\n", R1)
    print("Q1.T * Q1:\n", Q1.T @ Q1)
    print("Q1 * Q1.T:\n", Q1 @ Q1.T)
    print("QR:\n", Q1 @ R1)

    print()

    Q2, R2 = ModifiedGramSchmidt(A)
    print("Q2:\n", Q2)
    print("R2:\n", R2)
    print("Q2.T * Q2:\n", Q2.T @ Q2)
    print("Q2 * Q2.T:\n", Q2 @ Q2.T)
    print("QR:\n", Q2 @ R2)

    max_Q_diff = np.abs((Q1 - Q2)).max()
    max_R_diff = np.abs((R1 - R2)).max()
    if max_Q_diff != 0.0 and max_R_diff != 0.0:
        print(A)
        print(max_Q_diff, max_R_diff)
