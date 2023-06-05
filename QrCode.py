import numpy as np

def eigenvalues_qr(matrix, iterations=100):
    m, n = matrix.shape
    eigenvalues = []

    for _ in range(iterations):
        Q, R = np.linalg.qr(matrix)
        matrix = np.dot(R, Q)

    for i in range(n):
        eigenvalues.append(matrix[i, i])

    return eigenvalues

# Пример использования
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
eigenvalues = eigenvalues_qr(matrix)
print("Собственные числа:", eigenvalues)