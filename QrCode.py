import numpy as np


def eigenvalues_qr(A):
    n = A.shape[0]
    Q = np.copy(A)
    eigenvalues = []

    while True:
        Q, R = np.linalg.qr(Q)
        Q = np.dot(R, Q)

        off_diag_sum = np.sum(np.abs(Q - np.diag(np.diagonal(Q))))
        if off_diag_sum < 1e-6:
            break

    i = 0
    while i < n:
        if i == n - 1 or np.abs(Q[i + 1, i]) < 1e-6:
            # Реализация для некратных действительных или комплексных собственных значени
            eigenvalues.append(Q[i, i])
            i += 1
        else:
            # Реализация для комплексных собственных значений и кратности p
            lambda_1 = Q[i, i]
            lambda_2 = Q[i + 1, i + 1]
            eigenvalues.append(lambda_1 + lambda_2)
            eigenvalues.append(lambda_1 - lambda_2)
            i += 2

    return eigenvalues

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

eigenvalues = eigenvalues_qr(A)

for eigenvalue in eigenvalues:
    print(eigenvalue)
