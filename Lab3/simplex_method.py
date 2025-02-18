import numpy as np

def simplex_method(A, c, x_init, B_init, max_iter=1000, epsilon=1e-10):
    """
    Основная фаза симплекс-метода с возвратом базисных индексов.
    """
    m, n = A.shape
    B = [b - 1 for b in B_init]  # Переводим в 0-based
    x = x_init.copy().astype(float)
    
    for _ in range(max_iter):
        AB = A[:, B]
        try:
            AB_inv = np.linalg.inv(AB)
        except np.linalg.LinAlgError:
            return None, None
        
        cB = c[B]
        u = cB @ AB_inv
        
        delta = u @ A - c
        
        if np.all(delta >= -epsilon):
            return x, B
        
        j0 = np.argmax(delta < -epsilon)
        Aj0 = A[:, j0]
        z = AB_inv @ Aj0
        
        theta = []
        for i in range(m):
            ji = B[i]
            theta_i = x[ji] / z[i] if z[i] > epsilon else np.inf
            theta.append(theta_i)
        
        theta0 = min(theta)
        if np.isinf(theta0):
            return None, None
        
        k = np.argmin(theta)
        B[k] = j0
        
        x_new = x.copy()
        x_new[j0] = theta0
        for i in range(m):
            if i != k:
                x_new[B[i]] -= theta0 * z[i]
        x_new[B[k]] = 0.0
        x = x_new
    
    return x, B

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