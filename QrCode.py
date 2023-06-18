import numpy as np
from math import sqrt

epsilon = 1e-30

def read_square_matrix():
    size = int(input("Введите размерность квадратной матрицы: "))
    matrix = []
    print("Введите элементы матрицы построчно:")
    for _ in range(size):
        row = []
        for _ in range(size):
            element = int(input())
            row.append(element)
        matrix.append(row)
    return matrix, size


def qr_algorithm(A, size, max_iterations=100000):
    n = size

    for i in range(max_iterations):
        Q, R = np.linalg.qr(A)
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

        if np.abs(d)<epsilon:
            x1 = -b2 / 2
            eigenvalues.append(x1)

        else:
            x1 = complex(-b2 / 2, sqrt(np.abs(d)) / 2)  # как прописать в питоне эти мнимые числа как числа хз
            x2 = complex(-b2 / 2, -sqrt(np.abs(d)) / 2)
            eigenvalues.extend([x1, x2])


    if flag != 1:
        eigenvalues.append(A[-1][-1])

    return eigenvalues


if __name__ == "__main__":
    #A = np.array([[0, 0, 2],
    #              [1, 0, -5],
    #              [0, 1, 4]])

    A,size=read_square_matrix()

    Q, R = np.linalg.qr(A)

    print("Матрица Q:")
    print(Q)
    print("Матрица R:")
    print(R)

    print("Матрица A_res:")
    print(np.dot(Q, R))

    A_k = qr_algorithm(A,size)
    print("Матрица A_k:")
    print(A_k)

    print('eigenvalues:')
    for eigenvalue in find_eigenvalues(A_k):
        print(eigenvalue)

