import numpy as np


def input_matrix():
    n = int(input('Федя, ты заебал, введи рамерность матрицы: '))
    matrix = []
    nums = input('Ты будешь блять работать с комплексными или обычными числами. Напиши comlex или float: ').lower()
    com = True
    while True:
        if nums == 'complex':
            break
        elif nums == 'float':
            com = False
            break
        else:
            nums = input('Ты блять напиши complex or float: ')
    if com:
        print('Теперь вводи строки матрицы')
        for i in range(n):
            stroka1 = input('{} строка: '.format(i + 1)).split()
            stroka = []
            for c in stroka1:
                if '+' in c:
                    real = float(c[:c.index('+')])
                    mnim = float(c[c.index('+'): c.index('i')])
                elif '-' in c:
                    real = float(c[:c.index('-')])
                    mnim = float(c[c.index('-'): c.index('i')])
                else:
                    real = float(c)
                    mnim = 0
                stroka.append(complex(real, mnim))
            matrix.append(stroka)
    else:
        print('Теперь вводи строки матрицы')
        for i in range(n):
            stroka = list(map(float, input('{} строка: '.format(i + 1)).split()))
            matrix.append(stroka)
    return matrix


def qr_decomposition(matrix):
    m, n = matrix.shape
    q = np.eye(m, dtype=np.complex128)  # Единичная матрица комплексного типа
    r = matrix.astype(np.complex128).copy()  # Изменение типа данных на комплексный
    for j in range(n):
        # Применение матрицы отражения Хаусхолдера для преобразования столбца r[j:, j]
        x = r[j:, j]
        e = np.zeros_like(x)
        e[0] = 1
        v = np.sign(x[0]) * np.linalg.norm(x) * e + x
        v /= np.linalg.norm(v)

        v = v.reshape(-1, 1)  # Изменение формы v на вектор-столбец

        r[j:, j:] -= 2.0 * np.dot(v, np.dot(v.T.conjugate(), r[j:, j:]))  # Исправленная операция

        # Применение матрицы отражения Хаусхолдера для преобразования матрицы Q
        q[:, j:] -= 2.0 * np.dot(q[:, j:], np.dot(v, v.T.conjugate()))  # Исправленная операция
    return q, r


if __name__ == "__main__":
    matrix = np.array(input_matrix())
    print(matrix)
    # Пример использования
    # matrix = np.array([[1, 2, 3],
    # [4, 5, 6],
    # [7, 8, 9]])
    q, r = qr_decomposition(matrix)
    print("Матрица Q:")
    print(q)
    print("Матрица R:")
    print(r)
    print(np.dot(q, r))
