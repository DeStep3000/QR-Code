import numpy as np

def qr_decomposition(A):
    m, n = A.shape
    Q = np.eye(m)  # Инициализация матрицы Q как единичной матрицы размерности m x m
    R = A.copy()  # Копирование исходной матрицы A в матрицу R

    for j in range(n):
        # Построение матрицы отражения H_j
        x = R[j:, j]  # Выбираем столбец матрицы R, начиная с позиции j
        e = np.zeros_like(x)
        e[0] = 1  # Вектор e = (1, 0, ..., 0)

        u = np.sign(x[0]) * np.linalg.norm(x) * e + x  # Вектор u = sign(x[0]) * ||x|| * e + x
        u = u / np.linalg.norm(u)  # Нормализация вектора u

        # Построение матрицы отражения H_j = I - 2uu^T
        H = np.eye(m)
        H[j:, j:] -= 2.0 * np.outer(u, u)

        # Применение матрицы отражения H_j к матрицам Q и R
        Q = np.dot(Q, H.T)
        R = np.dot(H, R)

    return Q, R
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

Q, R = qr_decomposition(A)
res = np.dot(Q, R)

print("Матрица Q:")
print(Q)
print("Матрица R:")
print(R)
print("Матрица A:")
print(res)

import numpy as np


def qr_algorithm(A, max_iterations=100):
    n = A.shape[0]
    eigenvalues = np.zeros(n)

    for i in range(max_iterations):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

        # Проверка на сходимость
        if np.allclose(np.diag(A), eigenvalues):
            break

        eigenvalues = np.diag(A)

    return eigenvalues


A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

eigenvalues = qr_algorithm(A)
print("Собственные числа матрицы A:")
print(eigenvalues)


