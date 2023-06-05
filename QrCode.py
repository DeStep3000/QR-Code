import numpy as np


def find_eigenvalues(matrix, epsilon=1e-10, max_iterations=100):
    n = matrix.shape[0]
    eigenvalues = []
    iterations = 0

    while n > 1 and iterations < max_iterations:
        # QR-разложение матрицы
        q, r = np.linalg.qr(matrix)

        # Умножение R на Q для получения новой матрицы
        matrix = np.dot(r, q)

        # Поиск собственных значений на диагонали
        eigenvalues.extend(matrix.diagonal())

        # Проверка сходимости
        off_diagonal_sum = np.sum(np.abs(matrix) - np.abs(np.diag(matrix)))
        if off_diagonal_sum < epsilon:
            break

        n -= 1
        iterations += 1

    # Добавление последнего собственного значения
    eigenvalues.append(matrix[0, 0])

    return eigenvalues


# Пример использования
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

eigenvalues = find_eigenvalues(matrix)
print("Собственные значения:", eigenvalues)
