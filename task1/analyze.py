import os
import matplotlib.pyplot as plt

BINARY = "./task1"
TYPE = "column"

# Данные
processes = [1, 4, 8, 16]
sizes = [500, 1000, 2000, 4000]
data = []

for process in processes:
    values = []
    for size in sizes:
        print(f"Start {size} size for {process} threads")
        os.system(f"mpiexec -n {process} {BINARY} {size} {TYPE}")
        with open("time.txt") as file:
            time = float(file.readline())
            values.append(time)
            print(time)
    data.append(values)

# Создание графика
fig, ax = plt.subplots()
for i, row in enumerate(data):
    ax.plot(sizes, row, marker='o',
            label=f'{processes[i]} поток/ов')

# Настройка графика
title = "???"
if TYPE == "row":
    title = "разбиения по строкам"
if TYPE == "column":
    title = "разбиения по столбцам"
if TYPE == "block":
    title = "разбиения по блокам"
ax.set_title(f'Время выполнения умножения матрицы на вектор с помощью\n{title}')
ax.set_xlabel('Размер (N)')
ax.set_ylabel('Время выполнения (мс)')
ax.legend()
ax.grid()

# Отображение графика
plt.savefig(f'./statistic/{TYPE}2.png')
plt.show()