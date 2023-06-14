import numpy as np
from math import sqrt

epsilon = 1e-30


def qr_decomposition(A):
    m, n = A.shape
    Q = np.eye(m)  # Инициализация матрицы Q как единичной матрицы размером m x m
    R = np.copy(A)  # Создание копии исходной матрицы A

    for j in range(n):
        # Вычисление вектора отражения
        x = R[j:, j]
        norm_x = np.linalg.norm(x)
        v = np.zeros_like(x)
        v[0] = x[0] + np.copysign(norm_x, x[0])
        v[1:] = x[1:]

        # Применение преобразования Хаусхолдера к матрицам Q и R
        H = np.eye(m)
        H[j:, j:] -= 2.0 * np.outer(v, v) / np.dot(v, v)
        Q = np.dot(Q, H)
        R = np.dot(H, R)

    return Q, R


def qr_algorithm(A, max_iterations=100000):
    n = A.shape[0]

    for i in range(max_iterations):
        Q, R = qr_decomposition(A)
        A = np.dot(R, Q)

        # Проверяем условие сходимости
        off_diag_sum = np.sum(np.abs(A - np.diag(np.diagonal(A))))
        if off_diag_sum < epsilon:
            break
    return A


def find_eigenvalues(A):
    eigenvalues = []
    n = A.shape[0]
    flag = 0

    for i in range(n - 1):
        if flag == 1:
            flag = 0
            continue
        if abs(A[i + 1][i]) < epsilon:
            flag = 0
            eigenvalues.append(A[i][i])
            continue
        flag = 1
        a1 = A[i][i]
        b1 = A[i][i + 1]
        c1 = A[i + 1][i]
        d1 = A[i + 1][i + 1]

        a2 = 1
        b2 = -a1 - d1
        c2 = a1 * d1 - b1 * c1
        d = b2 ** 2 - 4 * a2 * c2

        if d == 0:
            x1 = -b2 / 2
            eigenvalues.append(x1)

        elif d < 0:
            x1 = complex(-b2 / 2, sqrt(-d) / 2)  # как прописать в питоне эти мнимые числа как числа хз
            x2 = complex(-b2 / 2, -sqrt(-d) / 2)
            eigenvalues.extend([x1, x2])
        else:
            print('Ошибка: дискриминант > 0')

    if flag != 1:
        eigenvalues.append(A[-1][-1])

    return eigenvalues


if __name__ == "__main__":
    A = np.array([[0, 0, 2],
                  [1, 0, -5],
                  [0, 1, 4]])

    Q, R = qr_decomposition(A)

    print("Матрица Q:")
    print(Q)
    print("Матрица R:")
    print(R)

    print("Матрица A_res:")
    print(np.dot(Q, R))

    A_k = qr_algorithm(A)
    print(A_k)
    eigenvalues_res = find_eigenvalues(A_k)

    print('eigenvalues:')
    for eigenvalue in eigenvalues_res:
        print(eigenvalue)
