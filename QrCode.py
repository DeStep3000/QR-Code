import numpy as np

def qr_decomposition(matrix):
    m, n = matrix.shape
    Q = np.eye(m)
    R = np.copy(matrix)

    for j in range(n):
        column = R[j:, j]
        norm = np.linalg.norm(column)
        sign = np.sign(column[0])
        v = column + sign * norm * np.eye(len(column))[:, 0]
        v = v / np.linalg.norm(v)

        R[j:, :] -= 2.0 * np.outer(v, np.dot(v.T, R[j:, :]))
        Q[:, j:] -= 2.0 * np.outer(Q[:, j:], np.dot(Q[:, j:].T, v))

    return Q, R

# Пример использования
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Q, R = qr_decomposition(matrix)
print("Матрица Q:")
print(Q)
print("Матрица R:")
print(R)