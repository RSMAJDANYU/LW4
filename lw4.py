import numpy as np
import matplotlib.pyplot as plt

# Ввод чисел K и N
K = int(input("Введите K: "))
N = int(input("Введите N: "))

# Создание матрицы A
np.random.seed(42)
A = np.random.randint(-10, 11, size=(N, N))

# Обработка случая N=1 отдельно
if N == 1:
    F = A.copy()
    # Для N=1 просто применяем второе условие по умолчанию
    det_A = A[0,0]
    sum_diag_F = F[0,0]
    
    if det_A > sum_diag_F:
        result = A @ A.T - K * (1/F[0,0] if F[0,0] != 0 else 0)
    else:
        result = (1/A[0,0] + A[0,0] - F[0,0]) * K if A[0,0] != 0 else 0
else:
    # Разделение матрицы на подматрицы
    half = N // 2
    B = A[:half, :half]
    C = A[:half, half+N%2:]
    D = A[half+N%2:, :half]
    E = A[half+N%2:, half+N%2:]

    # Проверка условия для подматрицы E
    zero_count = np.sum(E[:, ::2] == 0)
    product = np.prod(E[::2, :]) if E.size > 0 else 1

    # Создание матрицы F
    F = A.copy()

    if E.size > 0:  # Если подматрица E существует
        if zero_count * K > product:
            # Симметричный обмен B и C
            F[:half, :half] = C.T
            F[:half, half+N%2:] = B.T
        else:
            # Несимметричный обмен B и E
            F[:half, :half] = E
            F[half+N%2:, half+N%2:] = B

    # Проверка основного условия
    det_A = np.linalg.det(A)
    sum_diag_F = np.trace(F)

    if det_A > sum_diag_F:
        result = A @ A.T - K * np.linalg.inv(F)
    else:
        G = np.tril(A)
        result = (np.linalg.inv(A) + G - F.T) * K

# Вывод результатов
print("Матрица A:")
print(A)
print("\nМатрица F:")
print(F)
print("\nРезультат вычислений:")
print(result)

# Построение графиков (только для N > 1)
if N > 1:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(A, cmap='viridis')
    plt.colorbar()
    plt.title("Тепловая карта A")
    
    plt.subplot(1, 3, 2)
    plt.imshow(F, cmap='plasma')
    plt.colorbar()
    plt.title("Тепловая карта F")
    
    plt.subplot(1, 3, 3)
    plt.plot(np.diag(A), label='Диагональ A')
    plt.plot(np.diag(F), label='Диагональ F')
    plt.legend()
    plt.title("Сравнение диагоналей")
    
    plt.tight_layout()
    plt.show()
else:
    print("\nГрафики не строятся для N=1")