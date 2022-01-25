import numpy as np
import matplotlib.pyplot as plt


abc = np.array([
    [0, 0, 45, 7, 27, 0, 0, 0, 0, 1, 44, 0, 0, ],
    [0, 0, 13, 0, 0, 35, 0, 44, 0, 29, 0, 0, 37, ],
    [0, 27, 0, 14, 19, 0, 5, 0, 10, 27, 11, 0, 37, ],
    [41, 0, 0, 0, 0, 0, 0, 15, 12, 0, 0, 0, 48, ],
    [41, 0, 0, 0, 0, 0, 0, 15, 12, 0, 0, 0, 48, ],
    [0, 34, 0, 0, 28, 0, 0, 0, 0, 0, 0, 27, 0, ],
    [0, 0, 0, 27, 0, 2, 0, 10, 0, 30, 43, 34, 16, ],
    [0, 21, 13, 0, 0, 0, 0, 0, 9, 0, 0, 2, 0, ],
    [0, 33, 0, 0, 25, 0, 38, 0, 0, 37, 0, 0, 4, ],
    [0, 9, 9, 0, 28, 7, 0, 0, 17, 0, 5, 0, 0, ],
    [16, 27, 49, 0, 0, 10, 0, 16, 3, 0, 0, 15, 4, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, ],
    [0, 0, 0, 26, 0, 0, 0, 0, 0, 39, 11, 20, 0]
])

# Построениея графика по сумму строк матрицы
print('Построениея графика по сумму строк матрицы')
print(np.sum(abc, axis=1).tolist())
# Построениея графика по сумму столбцов матрицы
print('Построениe графика по сумму столбцов матрицы')
print(np.sum(abc, axis=0).tolist())
# Построениея графика по  среднему значению каждой строки
print(' Построениe графика по  среднему значению каждой строки')
print(np.average(abc, axis=1).tolist())
# Построениея графика по  среднему значению каждого столбца
print('Построениe графика по  среднему значению каждого столбца')
print(np.average(abc, axis=0).tolist())

plt.figure(1)
for row in range(len(np.sum(abc, axis=1))):
    for colums in range(len(np.sum(abc, axis=0))):
        plt.plot(row, colums, 'o', markersize=abc[row][colums])
plt.show()
