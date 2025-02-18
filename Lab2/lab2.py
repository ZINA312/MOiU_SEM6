import numpy as np

def simplex_method(A, c, x_init, B_init, max_iter=1000, epsilon=1e-10):
    """
    Основная фаза симплекс-метода для решения задачи линейного программирования.
    
    Параметры:
    A : numpy.ndarray
        Матрица коэффициентов размерности m x n
    c : numpy.ndarray
        Вектор коэффициентов целевой функции размерности n
    x_init : numpy.ndarray
        Начальный допустимый план размерности n
    B_init : list
        Список базисных индексов (1-based)
    max_iter : int
        Максимальное количество итераций
    epsilon : float
        Точность для проверки условий
        
    Возвращает:
    numpy.ndarray или None
        Оптимальный план или None, если задача не ограничена
    """
    m, n = A.shape
    # Преобразуем базисные индексы в 0-based
    B = [b - 1 for b in B_init]
    x = x_init.copy().astype(float)
    
    for _ in range(max_iter):
        # Шаг 1: Построение базисной матрицы и обращение
        AB = A[:, B]
        try:
            AB_inv = np.linalg.inv(AB)
        except np.linalg.LinAlgError:
            print("Ошибка: базисная матрица вырождена.")
            return None
        
        # Шаг 2: Формирование вектора cB
        cB = c[B]
        # Шаг 3: Вычисление вектора потенциалов(как переменные влияют)
        u = cB.T @ AB_inv
        
        # Шаг 4: Вычисление вектора оценок(насколько может улучшиться)
        delta = u @ A - c
        # Шаг 5: Проверка условия оптимальности
        if np.all(delta >= -epsilon):
            return x
        
        # Шаг 6: Поиск первого отрицательного элемента
        j0 = np.argmax(delta < -epsilon)

        # Шаг 7: Вычисление вектора z(изменение плана)
        Aj0 = A[:, j0]
        z = AB_inv @ Aj0
        
        # Шаг 8-9: Вычисление theta0
        theta = []
        for i in range(m):
            ji = B[i]
            zi = z[i]
            if zi > epsilon:
                theta_i = x[ji] / zi
            else:
                theta_i = np.inf
            theta.append(theta_i)
         
        theta0 = min(theta)
        # Шаг 10
        if np.isinf(theta0):
            print("Целевая функция не ограничена сверху")
            return None
        
        # Шаг 11: Нахождение индекса k
        k = np.argmin(theta)
        
        # Сохраняем старый базис
        B_old = B.copy()
        j_star = B_old[k]
        
        # Шаг 12: Обновление базиса
        B[k] = j0
        
        # Шаг 13: Обновление плана
        x_new = x.copy()
        x_new[j0] = theta0
        
        for i in range(m):
            if i != k:
                ji = B_old[i]
                x_new[ji] -= theta0 * z[i]
        
        x_new[j_star] = 0.0
        x = x_new
    
    print("Достигнуто максимальное число итераций")
    return x

# Пример использования
if __name__ == "__main__":
    # Пример из задания
    A = np.array([
        [-1, 1, 1, 0, 0],
        [ 1, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1]
    ], dtype=float)
    
    c = np.array([1, 1, 0, 0, 0], dtype=float)
    x_init = np.array([0, 0, 1, 3, 2], dtype=float)
    B_init = [3, 4, 5]  # 1-based индексы x3, x4, x5
    
    result = simplex_method(A, c, x_init, B_init)
    
    if result is not None:
        print("Оптимальный план:")
        print(np.round(result, 2))
    else:
        print("Решение не найдено")