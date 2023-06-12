import numpy as np

def qr_complex_eigenvalues(matrix, iterations=100):
    eigenvalues = []
    m = matrix.copy()

    for _ in range(iterations):
        q, r = np.linalg.qr(m)
        m = np.dot(r, q)

    i = 0
    while i < m.shape[0]:
        if np.iscomplex(m[i, i]):
            eigenvalues.append(m[i, i])
            eigenvalues.append(np.conjugate(m[i, i+1]))
            i += 2
        else:
            eigenvalues.append(m[i, i])
            i += 1

    return eigenvalues


# Пример использования
matrix = np.array([[0, 0, 2], [1, 0, -5], [0, 1, 4]])
eigenvalues = qr_complex_eigenvalues(matrix)
print("Собственные значения:", eigenvalues)