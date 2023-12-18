import os
import matplotlib.pyplot as plt

BINARY = "task2/task2"

# Данные
processes = [1, 2, 4, 6, 8]
sizes = [100, 250, 500, 1000]
data = []

for process in processes:
    values = []
    for size in sizes:
        print(f"Start {size} size for {process} threads")
        os.system(f"mpiexec -n {process} {BINARY} {size}")
        with open("task2/time.txt") as file:
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
ax.set_title(f'Время выполнения умножения матрицы на вектор\n с помощью алгоритма Кэннона')
ax.set_xlabel('Размер (N)')
ax.set_ylabel('Время выполнения (мс)')
ax.legend()
ax.grid()

# Отображение графика
plt.savefig(f'task2/statistic/graph.png')
plt.show()