import numpy as np


def Givens(A):
    rows, cols = A.shape

    R = A.copy()
    Q = np.eye(rows)

    for j in range(0, cols):
        for i in reversed(range(j + 1, rows)):
            x = R[i - 1, j]
            y = R[i, j]
            if y == 0.0:  # already zero'd out
                continue
            else:  # y != 0
                theta = np.arctan2(y, x)
                c = np.cos(theta)
                s = np.sin(theta)
                G = np.eye(rows)
                G[i - 1, i - 1] = c
                G[i - 1, i] = s
                G[i, i - 1] = -s
                G[i, i] = c
                R = G @ R
                Q = Q @ G.T
                # print(i, j, 57 * theta)
                # print(f"G: {G}")
                # print(f"R: {R}")
                # print()
    return Q, R


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    A = np.array(
        [
            [1, 2, 3],
            [0, 0, 0],
            [-1, 0, 10],
        ]
    )
    print(np.linalg.det(A))
    Q, R = Givens(A)
    print(f"Q: {Q}")
    print(f"R: {R}")
    print(f"Recovered A:\n {Q@R}")
