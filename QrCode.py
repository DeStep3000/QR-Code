import numpy as np

def read_matrix():
    # Получить размерность матрицы от пользователя
    n = int(input("Введите размерность матрицы: "))

    # Создать пустую матрицу заданной размерности
    matrix = np.zeros((n, n), dtype=np.complex128)

    # Ввод элементов матрицы
    for i in range(n):
        for j in range(n):
            print(f"Введите элемент [{i}, {j}]: ")
            real_part = float(input("Вещественная часть: "))
            imag_part = float(input("Мнимая часть: "))
            matrix[i, j] = real_part + imag_part * 1j

    return matrix


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
            # Реализация для некратных действительных или комплексных собственных значений
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

# Считывание матрицы с клавиатуры
A = read_matrix()

# Нахождение собственных чисел
eigenvalues = eigenvalues_qr(A)

# Вывод собственных чисел
for eigenvalue in eigenvalues:
    print(eigenvalue)
