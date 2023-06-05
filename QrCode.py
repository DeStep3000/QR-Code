import numpy as np

def qr_decomposition(matrix):
    m, n = matrix.shape
    q = np.eye(m)  # Единичная матрица
    r = matrix.astype(np.float64).copy()  # Изменение типа данных на float64

    for j in range(n):
        # Применение матрицы отражения Хаусхолдера для преобразования столбца r[j:, j]
        x = r[j:, j]
        e = np.zeros_like(x)
        e[0] = 1
        v = np.sign(x[0]) * np.linalg.norm(x) * e + x
        v /= np.linalg.norm(v)

        v = v.reshape(-1, 1)  # Изменение формы v на вектор-столбец

        r[j:, j:] -= 2.0 * np.dot(v, np.dot(v.T, r[j:, j:]))  # Исправленная операция

        # Применение матрицы отражения Хаусхолдера для преобразования матрицы Q
        q[:, j:] -= 2.0 * np.dot(q[:, j:], np.dot(v, v.T))  # Исправленная операция

    return q, r

# Пример использования
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

q, r = qr_decomposition(matrix)
print("Матрица Q:")
print(q)
print("Матрица R:")
print(r)
