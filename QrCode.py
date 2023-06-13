import numpy as np

epsilon=1e-10
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

# Пример использования
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

Q, R = qr_decomposition(A)
A_res= np.dot(Q, R)

print("Матрица Q:")
print(Q)
print("Матрица R:")
print(R)
A_res= np.dot(Q, R)
print("Матрица A_res:")
print(A_res)

def qr_algorithm(A, max_iterations=1000):
    n = A.shape[0]

    for i in range(max_iterations):
        Q, R = qr_decomposition(A)
        A = np.dot(R, Q)

        # Проверяем условие сходимости
        off_diag_sum = np.sum(np.abs(A - np.diag(np.diagonal(A))))
        if off_diag_sum < epsilon:
            break

    return A

A_k=qr_algorithm(A)

def find_eigenvalues(A):
    eigenvalues = []
    n = A.shape[0]
    flag=0

    for i in range(n-1):
        if flag==1:
            flag=0
            continue
        if abs(A[i+1][i])<epsilon:
            flag = 0
            eigenvalues.append(A[i][i])
            continue

        a1=A[i][i]
        b1=A[i][i+1]
        c1=A[i+1][i]
        d1=A[i+1][i+1]

        a2=1
        b2=-a1-d1
        c2=a1*d1-b1*c1
        d=b2*b2-4*a2*c2

        if d==0:
            x1=-b2/2
            eigenvalues.append(x1)

        if d<0:
            x1=-b2/2+sqrt(-d)*IMAG /2#как прописать в питоне эти мнимые числа как числа хз
            x1 = -b2 / 2 - sqrt(-d)*IMAG / 2

    return eigenvalues

eigenvalues_res = find_eigenvalues(A)

print('eigenvalues:')
for eigenvalue in eigenvalues_res:
    print(eigenvalue)