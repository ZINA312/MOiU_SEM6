from collections import Counter
import numpy as np

def balance(a, b, c):
    total_a = np.sum(a)
    total_b = np.sum(b)
    if total_a > total_b:
        b = np.hstack((b, np.array([total_a - total_b])))
        c = np.hstack((c, np.zeros((c.shape[0], 1))))
    elif total_a < total_b:
        a = np.hstack((a, np.array([total_b - total_a])))
        c = np.hstack((c, np.zeros((c.shape[0], 1))))
    return a, b, c

def solve_transportation_problem(a, b, c):
    # Балансировка задачи
    a, b, c = balance(a, b, c)
    m, n = c.shape
    X = np.zeros_like(c)
    B = []

    # Фаза 1: Построение начального плана
    i, j = 0, 0
    while i < m and j < n:
        minimum = min(a[i], b[j])
        X[i, j] = minimum
        B.append((i, j))
        a[i] -= minimum
        b[j] -= minimum
        if a[i] == 0:
            i += 1
        else:
            j += 1    
    
    # Фаза 2: Оптимизация плана
    while True:
        # Построение системы уравнений
        A = np.zeros((m + n, m + n))
        b_eq = np.zeros(m + n)
        for idx, (i, j) in enumerate(B):
            A[idx, i] = 1
            A[idx, m + j] = 1
            b_eq[idx] = c[i, j]
        A[-1, 0] = 1  
        
        try:
            u_v = np.linalg.solve(A, b_eq)
        except np.linalg.LinAlgError:
            u_v = np.linalg.lstsq(A, b_eq, rcond=None)[0]
        
        u, v = u_v[:m], u_v[m:]
        
        # Проверка оптимальности (все оценки неположительны)
        optimal = True
        entering = None
        for i in range(m):
            for j in range(n):
                if (i, j) not in B and u[i] + v[j] > c[i, j]:
                    optimal = False
                    entering = (i, j)
                    break
            if not optimal:
                B.append(entering)
                break
        
        if optimal:
            break
        
        # Поиск цикла
        B_copy = B.copy()
        while True:
            i_counts = Counter([bi for bi, _ in B_copy])
            j_counts = Counter([bj for _, bj in B_copy])
            i_to_remove = [bi for bi, cnt in i_counts.items() if cnt == 1]
            j_to_remove = [bj for bj, cnt in j_counts.items() if cnt == 1]
            if not i_to_remove and not j_to_remove:
                break
            B_copy = [(bi, bj) for bi, bj in B_copy if bi not in i_to_remove and bj not in j_to_remove]
        
        # Формирование цикла
        cycle = [B_copy.pop()]
        while B_copy:
            last = cycle[-1]
            found = False
            for idx, (bi, bj) in enumerate(B_copy):
                if bi == last[0] or bj == last[1]:
                    cycle.append(B_copy.pop(idx))
                    found = True
                    break
            if not found:
                break
        
        # Обновление плана
        plus = cycle[::2]
        minus = cycle[1::2]
        theta = min(X[bi, bj] for bi, bj in minus)
        
        for bi, bj in plus:
            X[bi, bj] += theta
        for bi, bj in minus:
            X[bi, bj] -= theta
        
        # Удаление нулевой базисной переменной
        zero_pos = next((bi, bj) for bi, bj in minus if X[bi, bj] == 0)
        B.remove(zero_pos)
    
    return X

if __name__ == "__main__":
    a = np.array([100, 300, 300])
    b = np.array([300, 200, 200])
    c = np.array([[8, 4, 1],
                  [8, 4, 3],
                  [9, 7, 5]])
    
    optimal = solve_transportation_problem(a, b, c)
    
    print("Оптимальная матрица перевозок:")
    print(optimal)
    print("\nДетализация перевозок:")
    for i in range(optimal.shape[0]):
        for j in range(optimal.shape[1]):
            if optimal[i, j] > 0:
                print(f"От поставщика {i+1} к потребителю {j+1}: {int(optimal[i, j])} единиц")