# Импорт необходимых модулей
import numpy as np
from math import sqrt

# Установка значения epsilon для проверки условия сходимости
epsilon = 1e-30


# Функция для чтения квадратной матрицы из пользовательского ввода
def read_square_matrix():
    size = int(input("Введите размерность квадратной матрицы: "))  # Запрашиваем размерность матрицы у пользователя
    matrix = []
    print("Введите элементы матрицы:")
    for i in range(size):
        print('{} строка'.format(i + 1))  # Запрашиваем элементы строки от пользователя
        row = []
        for _ in range(size):
            element = int(input())  # Запрашиваем элемент матрицы от пользователя
            row.append(element)
        matrix.append(row)
    return matrix, size


# Функция для выполнения QR-разложения матрицы
def qr_decomposition(A, size):
    m = size
    n = size
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


# Функция для выполнения QR-алгоритма
def qr_algorithm(A, size, max_iterations=100000):
    for i in range(max_iterations):
        Q, R = qr_decomposition(A, size)
        A = np.dot(R, Q)

        # Проверяем условие сходимости
        off_diag_sum = np.sum(np.abs(A - np.diag(np.diagonal(A))))
        if off_diag_sum < epsilon:
            break
    return A


# Функция для нахождения собственных значений матрицы
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

        if np.abs(d) < epsilon:
            x1 = -b2 / 2
            eigenvalues.append(x1)

        else:
            x1 = complex(-b2 / 2, sqrt(np.abs(d)) / 2)
            x2 = complex(-b2 / 2, -sqrt(np.abs(d)) / 2)
            eigenvalues.extend([x1, x2])

    if flag != 1:
        eigenvalues.append(A[-1][-1])

    return eigenvalues


# Основной код программы
if __name__ == "__main__":
    # A = np.array([[0, 0, 2],
    #              [1, 0, -5],
    #              [0, 1, 4]])

    A, size = read_square_matrix()

    Q, R = qr_decomposition(A, size)

    print("Матрица Q:")
    print(Q)
    print("Матрица R:")
    print(R)

    print("Матрица A_res:")
    print(np.dot(Q, R))

    A_k = qr_algorithm(A, size)
    print("Матрица A_k:")
    print(A_k)

    print('Собственные значения:')
    for eigenvalue in find_eigenvalues(A_k):
        print(eigenvalue)
