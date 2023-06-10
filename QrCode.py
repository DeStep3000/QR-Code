import numpy as np

def read_matrix():
    matrix = []
    size = int(input("Введите размер квадратной матрицы: "))

    print("Введите элементы матрицы:")

    for i in range(size):
        row = []
        for j in range(size):
            element = int(input(f"Введите элемент [{i+1}, {j+1}]: "))
            row.append(element)
        matrix.append(row)

    alpha= int(input('Введите 1, если собственные значения вещественные, 2-комлексно-сопряженные, 3-кратные, 4-комплексно-сопряженнные кратные: '))
    p=0
    if (alpha==3 or alpha==4):
        p=int(input('Введите число, которому собственные значения кратны: '))

    return matrix, alpha,p

def eigenvalues_qr(matrix, alpha,p, iterations=100):
    m, n = matrix.shape
    eigenvalues = []

    for _ in range(iterations):
        Q, R = np.linalg.qr(matrix)
        matrix = np.dot(R, Q)

    if alpha==1:
        for i in range(n):
            eigenvalues.append(matrix[i, i])

    if alpha == 2:
        i=n-2
        eigenvalues.append(matrix[i:,i:])

    if alpha == 3:
        i=n-p
        eigenvalues.append(matrix[i:, i:])

    if alpha == 4:
        i = n - 2*p
        eigenvalues.append(matrix[i:, i:])
    return eigenvalues

# Пример использования
#matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix,alpha,p = read_matrix()
print(matrix)
eigenvalues = eigenvalues_qr(matrix,alpha,p)
print("Собственные числа:", eigenvalues)