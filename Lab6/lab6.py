import numpy as np

def solve_qp():
    # Коэффициенты целевой функции: 1/2 x^T D x + c^T x
    D = np.array([
        [4., 2., 2., 0.],
        [2., 2., 0., 0.],
        [2., 0., 2., 0.],
        [0., 0., 0., 0.]
    ])
    c = np.array([-8., -6., -4., -6.])
    
    # Ограничения равенств: A_eq x = b_eq
    A_eq = np.array([
        [1., 0., 2., 1.],
        [0., 1., -1., 2.]
    ])
    b_eq = np.array([2., 3.])

    # Формируем расширенные ограничения
    A_active = np.vstack((
        A_eq,
        [0., 0., 1., 0.],  # x3=0
        [0., 0., 0., 1.]   # x4=0
    ))
    b_active = np.hstack((b_eq, [0., 0.]))
    
    # Решаем систему для условий стационарности
    n = D.shape[0]
    m = A_active.shape[0]
    
    # Матрица системы для условий ККТ
    KKT = np.block([
        [D,          A_active.T],
        [A_active,   np.zeros((m, m))]
    ])
    
    # Вектор правой части
    rhs = np.hstack((-c, b_active))
    
    # Решение системы
    solution = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    
    x = solution[:n]
    lambda_mu = solution[n:]
    
    # Проверка допустимости
    if np.all(x >= -1e-6) and np.allclose(A_eq @ x, b_eq):
        print("Найдено решение:")
        print(f"x1 = {x[0]:.4f}")
        print(f"x2 = {x[1]:.4f}")
        print(f"x3 = {x[2]:.4f}")
        print(f"x4 = {x[3]:.4f}")
    else:
        print("Решение не удовлетворяет ограничениям")

solve_qp()