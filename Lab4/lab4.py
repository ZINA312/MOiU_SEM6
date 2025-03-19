import numpy as np

def dual_simplex(c, A, b, initial_basis, max_iter=100):
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape  # Размеры матрицы A (количество строк и столбцов)
    B = np.array(initial_basis, dtype=int)  # Индексы базисных переменных
    
    for itter in range(max_iter):
        # Шаг 1: Формируем базисную матрицу A_B  и обратную
        AB = A[:, B]
        try:
            AB_inv = np.linalg.inv(AB)  
        except np.linalg.LinAlgError:
            return "Ошибка: вырожденная матрица AB" 
        
        # Шаг 2: Определяем вектор коэффициентов целевой функции для базисных переменных (c_B)
        cB = c[B]
        
        # Шаг 3: Вычисляем двойственные переменные (y = c_B * AB^(-1))
        y = cB @ AB_inv
        
        # Шаг 4: Находим псевдоплан (kappa_B = AB^(-1) * b)
        kappa_B = AB_inv @ b
        kappa = np.zeros(n) 
        kappa[B] = kappa_B  
        print(itter)
        # Шаг 5: Проверяем оптимальность псевдоплана
        if np.all(kappa >= -1e-10):  
            kappa = np.round(kappa, 6)  
            kappa[kappa < 0] = 0    
            return kappa 
        
        # Шаг 6: Выбор исключаемой переменной
        min_val = np.inf
        k = -1
        for i in range(len(kappa_B)):
            if kappa_B[i] < min_val: 
                min_val = kappa_B[i]
                k = i 
        
        # Шаг 7: Вычисляем вектор Δy
        delta_y = AB_inv[k, :]
        
        # Шаг 8: Проверка на несовместность двойственной задачи
        non_basis = [j for j in range(n) if j not in B]  
        mu = np.array([delta_y @ A[:, j] for j in non_basis])  
        
        if np.all(mu >= -1e-10):  
            return "Прямая задача не совместна"
        
        # Шаги 9-10: Вычисляем σ для небазисных переменных и выбираем включаемую переменную
        sigma = []
        valid_non_basis = []
        for idx, j in enumerate(non_basis):
            if mu[idx] < -1e-10: 
                Aj = A[:, j]  
                numerator = c[j] - Aj @ y  
                sigma_j = numerator / mu[idx]  
                sigma.append(sigma_j)
                valid_non_basis.append(j)  
            else:
                sigma.append(np.inf)  
        
        if not valid_non_basis: 
            return "Прямая задача не совместна"
        
        # Находим переменную с минимальным σ 
        min_sigma = np.inf
        j0 = -1
        for idx, j in enumerate(valid_non_basis):
            if sigma[idx] < min_sigma:
                min_sigma = sigma[idx]
                j0 = j

        # Шаг 11: Обновляем базис (заменяем исключаемую переменную на включаемую)
        B[k] = j0
    
    return "Достигнут максимум итераций"

c = [-4, -3, -7, 0, 0]  # Коэффициенты целевой функции
A = np.array([
    [-2, -1, -4, 1, 0],
    [-2, -2, -2, 0, 1]
], dtype=float)  # Матрица ограничений
b = np.array([-1, -1.5], dtype=float)  # Правая часть ограничений
initial_basis = [3, 4]  # Базисные индексы x4 и x5 (0-based)

result = dual_simplex(c, A, b, initial_basis)
print("Оптимальный план:", result)