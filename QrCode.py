import numpy as np


def qr_eigenvalues(matrix, iterations=100):
    n = matrix.shape[0]
    eigenvalues = np.zeros(n)
    for _ in range(iterations):
        Q, R = np.linalg.qr(matrix)
        matrix = np.dot(R, Q)

    for i in range(n):
        eigenvalues[i] = matrix[i, i]

    return eigenvalues


# Пример использования
# Создаем квадратную матрицу
A = np.array([[3, -1, 0], [-1, 2, -1], [0, -1, 1]])

# Находим собственные числа
eigenvalues = qr_eigenvalues(A)

# Выводим результаты
print("Собственные числа матрицы A:")
print(eigenvalues)