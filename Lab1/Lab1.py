import numpy as np

def invert_matrix_with_modified_column(A, x, i):
    #Находим A^-1 и вычисляем l = A^-1 * x
    A_inv = np.linalg.inv(A)
    l = A_inv @ x

    #Проверяем, является ли матрица A обратимой
    if l[i] == 0:
        return None, False 
    
    #Формируем вектор l_e
    l_e = l.copy()
    l_e[i] = -1
    
    #Находим ˆl
    hat_l = -l_e / l[i]
    
    #Формируем матрицу Q
    n = A.shape[0]
    Q = np.eye(n)
    Q[:, i] = hat_l 
    
    #Находим (A)^-1 = QA^-1
    A_inv_modified = Q @ A_inv
    
    return A_inv_modified, True

A = np.array([[1, -1, 0],
              [0, 1, 0],
              [0, 0, 1]
            ])
x = np.array([1, 0, 1])
i = 2

A_inv, is_invertible = invert_matrix_with_modified_column(A, x, i)

if is_invertible:
    print("Обратная матрица A с измененным столбцом:")
    print(A_inv)
else:
    print("Матрица A необратима.")