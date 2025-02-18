import numpy as np
from simplex_method import simplex_method

def initial_phase_simplex(A, c, b, max_iter=1000, epsilon=1e-10):
    # Шаг 1: Делаем b неотрицательным
    m, n = A.shape
    for i in range(m):
        if b[i] < 0:
            A[i] *= -1
            b[i] *= -1

    # Шаг 2: Создаем вспомогательную задачу
    aux_A = np.hstack([A, np.eye(m)])
    aux_c = np.hstack([np.zeros(n), -np.ones(m)])
    aux_n = n + m

    # Шаг 3: Начальный базисный план
    x_aux_init = np.zeros(aux_n)
    x_aux_init[n:] = b.copy()
    B_aux_init = [n + i + 1 for i in range(m)]

    # Шаг 4: Решаем вспомогательную задачу
    x_aux, B_aux = simplex_method(aux_A, aux_c, x_aux_init, B_aux_init, max_iter, epsilon)
    if x_aux is None:
        print("Вспомогательная задача не имеет решения")
        return None

    # Шаг 5: Проверяем искусственные переменные
    if not np.allclose(x_aux[n:], 0, atol=epsilon):
        print("Исходная задача несовместна")
        return None

    # Шаг 6: Формируем допустимый план
    x = x_aux[:n].copy()
    current_A = aux_A.copy()
    current_b = b.copy()
    current_B = [j for j in B_aux]  

    # Корректируем базис
    while True:
        artificial = [j for j in current_B if j >= n]
        if not artificial:
            break
        jk = max(artificial)
        k_pos = current_B.index(jk)
        i = jk - n

        # Поиск замены
        AB = current_A[:, current_B]
        try:
            AB_inv = np.linalg.inv(AB)
        except np.linalg.LinAlgError:
            AB_inv = None

        non_basis = [j for j in range(n) if j not in current_B]
        found = False
        if AB_inv is not None:
            for j in non_basis:
                Aj = current_A[:, j]
                lj = AB_inv @ Aj
                if not np.isclose(lj[k_pos], 0, atol=epsilon):
                    current_B[k_pos] = j
                    found = True
                    break
        if not found:
            if i >= current_A.shape[0]:
                return None
            current_A = np.delete(current_A, i, axis=0)
            current_b = np.delete(current_b, i)
            current_B.pop(k_pos)
            for idx in range(len(current_B)):
                j = current_B[idx]
                if j >= n:
                    orig_i = j - n
                    if orig_i > i:
                        current_B[idx] = n + (orig_i - 1)

    if len(current_B) != current_A.shape[0]:
        return None

    A_updated = current_A[:, :n]
    B_updated = [jb + 1 for jb in current_B]  

    return x, B_updated, A_updated, current_b

# Пример использования
if __name__ == "__main__":
    A = np.array([[1, 1, 1], [2, 2, 2]], dtype=float)
    c = np.array([1, 0, 0], dtype=float)
    b = np.array([0, 0], dtype=float)

    result = initial_phase_simplex(A, c, b)
    if result:
        x, B, A_new, b_new = result
        print("Базисный план:", x)
        print("Базисные индексы:", B)
        print("Обновленная A:\n", A_new)
        print("Обновленный b:", b_new)
    else:
        print("Задача несовместна")