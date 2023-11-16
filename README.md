<details>
<summary>Лаб 1</summary>

# Постановка задачи
Написать многослойный перспетрон и обучить нейронную сеть на датасете MNIST. Визуализировать процесс обучения и исследовать зависимость точности от размера batch
# Вывод результата

Training with batch size: 16
Epoch 1/5 - Train loss: 1696.4849, Train acc: 0.8732, Test loss: 44.0088, Test acc: 0.9189
Epoch 2/5 - Train loss: 959.4164, Train acc: 0.9252, Test loss: 34.1600, Test acc: 0.9353
Epoch 3/5 - Train loss: 741.6987, Train acc: 0.9426, Test loss: 27.0549, Test acc: 0.9500
Epoch 4/5 - Train loss: 607.4574, Train acc: 0.9535, Test loss: 22.4875, Test acc: 0.9569
Epoch 5/5 - Train loss: 514.6806, Train acc: 0.9602, Test loss: 20.4098, Test acc: 0.9630
Training with batch size: 32
Epoch 1/5 - Train loss: 1067.8972, Train acc: 0.8502, Test loss: 51.2176, Test acc: 0.9096
Epoch 2/5 - Train loss: 598.0913, Train acc: 0.9073, Test loss: 44.8074, Test acc: 0.9202
Epoch 3/5 - Train loss: 515.8500, Train acc: 0.9205, Test loss: 38.7503, Test acc: 0.9294
Epoch 4/5 - Train loss: 452.0708, Train acc: 0.9306, Test loss: 34.2029, Test acc: 0.9355
Epoch 5/5 - Train loss: 398.2344, Train acc: 0.9392, Test loss: 31.1015, Test acc: 0.9414
Training with batch size: 64
Epoch 1/5 - Train loss: 691.9277, Train acc: 0.8184, Test loss: 61.5366, Test acc: 0.8909
Epoch 2/5 - Train loss: 344.4888, Train acc: 0.8953, Test loss: 51.0744, Test acc: 0.9056
Epoch 3/5 - Train loss: 302.2391, Train acc: 0.9066, Test loss: 46.3142, Test acc: 0.9161
Epoch 4/5 - Train loss: 278.3382, Train acc: 0.9146, Test loss: 44.6688, Test acc: 0.9167
Epoch 5/5 - Train loss: 259.9853, Train acc: 0.9210, Test loss: 40.6324, Test acc: 0.9280
Training with batch size: 128
Epoch 1/5 - Train loss: 489.9462, Train acc: 0.7682, Test loss: 81.2860, Test acc: 0.8720
Epoch 2/5 - Train loss: 214.7820, Train acc: 0.8776, Test loss: 61.1081, Test acc: 0.8946
Epoch 3/5 - Train loss: 178.6892, Train acc: 0.8931, Test loss: 53.5759, Test acc: 0.9037
Epoch 4/5 - Train loss: 162.8944, Train acc: 0.9006, Test loss: 49.7520, Test acc: 0.9090
Epoch 5/5 - Train loss: 153.1088, Train acc: 0.9061, Test loss: 47.9675, Test acc: 0.9132

# Визуализация



# Заключение

В коде определен простой MLP с одним скрытым слоем. Этот слой имеет 128 нейронов и функцию активации ReLU, а выходной слой имеет 10 нейронов для предсказания классов

Нейронная сеть обучается на датасете MNIST с использованием стохастического градиентного спуска (SGD) и функции потерь CrossEntropyLoss

Процесс обучения визуализируется для разных размеров batch. Видно, как изменение размера batch влияет на точность модели по мере обучения

Из графика видно, что оптимальный размер batch может зависеть от конкретной задачи. Иногда использование маленького batch может привести к более стабильному обучению, но требует больше времени на обработку данных. В то время как больший batch может ускорить обучение, но может снизить его стабильность
</details>

<details>
<summary>Лаб 2</summary>
# Постановка задачи
В рамках выполненияя лабораторной работы необходимо:

Реализовать многослойный персептрон – feedforward сеть с полносвязными слоями для классификации изображений из датасета CIFAR, провести эксперименты по визуализации процесса обучения и исследованию зависимости точности от размера batch.

# Вывод результата

Training with batch size: 16
Epoch 1/10 - Train loss: 1.6906, Train acc: 0.3976, Test loss: 1.5851, Test acc: 0.4421
Epoch 2/10 - Train loss: 1.5076, Train acc: 0.4687, Test loss: 1.5447, Test acc: 0.4631
Epoch 3/10 - Train loss: 1.4121, Train acc: 0.5037, Test loss: 1.4779, Test acc: 0.4844
Epoch 4/10 - Train loss: 1.3369, Train acc: 0.5306, Test loss: 1.4612, Test acc: 0.4964
Epoch 5/10 - Train loss: 1.2695, Train acc: 0.5530, Test loss: 1.4824, Test acc: 0.4917
Epoch 6/10 - Train loss: 1.2143, Train acc: 0.5718, Test loss: 1.5092, Test acc: 0.4924
Epoch 7/10 - Train loss: 1.1557, Train acc: 0.5914, Test loss: 1.4782, Test acc: 0.5034
Epoch 8/10 - Train loss: 1.1109, Train acc: 0.6060, Test loss: 1.4842, Test acc: 0.5168
Epoch 9/10 - Train loss: 1.0631, Train acc: 0.6245, Test loss: 1.5107, Test acc: 0.5198
Epoch 10/10 - Train loss: 1.0097, Train acc: 0.6414, Test loss: 1.5788, Test acc: 0.5165
Training with batch size: 32
Epoch 1/10 - Train loss: 0.8827, Train acc: 0.6853, Test loss: 1.5826, Test acc: 0.5238
Epoch 2/10 - Train loss: 0.8168, Train acc: 0.7077, Test loss: 1.6542, Test acc: 0.5205
Epoch 3/10 - Train loss: 0.7728, Train acc: 0.7226, Test loss: 1.7292, Test acc: 0.5235
Epoch 4/10 - Train loss: 0.7297, Train acc: 0.7399, Test loss: 1.8150, Test acc: 0.5251
Epoch 5/10 - Train loss: 0.6993, Train acc: 0.7500, Test loss: 1.8498, Test acc: 0.5213
Epoch 6/10 - Train loss: 0.6712, Train acc: 0.7601, Test loss: 1.8816, Test acc: 0.5197
Epoch 7/10 - Train loss: 0.6386, Train acc: 0.7717, Test loss: 1.9910, Test acc: 0.5218
Epoch 8/10 - Train loss: 0.6116, Train acc: 0.7806, Test loss: 2.0495, Test acc: 0.5135
Epoch 9/10 - Train loss: 0.5835, Train acc: 0.7910, Test loss: 2.1808, Test acc: 0.5217
Epoch 10/10 - Train loss: 0.5688, Train acc: 0.7958, Test loss: 2.2933, Test acc: 0.5247
Training with batch size: 64
Epoch 1/10 - Train loss: 0.4748, Train acc: 0.8290, Test loss: 2.4010, Test acc: 0.5266
Epoch 2/10 - Train loss: 0.4262, Train acc: 0.8437, Test loss: 2.6004, Test acc: 0.5275
Epoch 3/10 - Train loss: 0.4123, Train acc: 0.8506, Test loss: 2.5886, Test acc: 0.5231
Epoch 4/10 - Train loss: 0.3908, Train acc: 0.8587, Test loss: 2.7805, Test acc: 0.5155
Epoch 5/10 - Train loss: 0.3825, Train acc: 0.8614, Test loss: 2.8284, Test acc: 0.5143
Epoch 6/10 - Train loss: 0.3673, Train acc: 0.8652, Test loss: 2.7998, Test acc: 0.5261
Epoch 7/10 - Train loss: 0.3472, Train acc: 0.8750, Test loss: 2.9602, Test acc: 0.5305
Epoch 8/10 - Train loss: 0.3500, Train acc: 0.8740, Test loss: 2.9545, Test acc: 0.5149
Epoch 9/10 - Train loss: 0.3263, Train acc: 0.8822, Test loss: 3.1395, Test acc: 0.5251
Epoch 10/10 - Train loss: 0.3288, Train acc: 0.8825, Test loss: 3.2085, Test acc: 0.5093
Training with batch size: 128
Epoch 1/10 - Train loss: 0.2638, Train acc: 0.9044, Test loss: 3.4332, Test acc: 0.5271
Epoch 2/10 - Train loss: 0.2325, Train acc: 0.9161, Test loss: 3.5790, Test acc: 0.5244
Epoch 3/10 - Train loss: 0.2352, Train acc: 0.9160, Test loss: 3.6567, Test acc: 0.5300
Epoch 4/10 - Train loss: 0.2137, Train acc: 0.9227, Test loss: 3.7563, Test acc: 0.5255
Epoch 5/10 - Train loss: 0.2193, Train acc: 0.9207, Test loss: 3.9140, Test acc: 0.5268
Epoch 6/10 - Train loss: 0.2129, Train acc: 0.9225, Test loss: 3.9716, Test acc: 0.5166
Epoch 7/10 - Train loss: 0.2161, Train acc: 0.9226, Test loss: 3.9912, Test acc: 0.5247
Epoch 8/10 - Train loss: 0.1998, Train acc: 0.9279, Test loss: 4.1420, Test acc: 0.5184
Epoch 9/10 - Train loss: 0.1937, Train acc: 0.9296, Test loss: 4.2752, Test acc: 0.5320
Epoch 10/10 - Train loss: 0.1878, Train acc: 0.9311, Test loss: 4.3741, Test acc: 0.5237

# Визуализация



# Заключение

Многослойный персептрон (MLP): Определен для классификации изображений CIFAR-10 с использованием PyTorch. В реализации используется два полносвязанных слоя:

Первый слой (входной слой):
Размер входных данных: 32 * 32 * 3 (ширина * высота * количество каналов)
Полносвязанный слой (nn.Linear) с 32 * 32 * 3 входами и 512 выходами

Второй слой (выходной слой):
Полносвязанный слой с 512 входами и 10 выходами (по числу классов в CIFAR-10)

Обучение на CIFAR-10: Нейронная сеть обучается на датасете CIFAR-10 с использованием оптимизатора Adam и функции потерь CrossEntropyLoss

Оптимизатор Adam является адаптивным оптимизатором, который пытается эффективно настраивать скорость обучения для каждого параметра в модели.
Функция потерь CrossEntropyLoss используется в задачах классификации, когда модель предсказывает вероятности принадлежности каждого класса

Исследование зависимости от размера batch: Проведен эксперимент по визуализации процесса обучения для разных размеров batch, визуализации изображений из тестового набора и исследованию зависимости точности от размера batch. На этом графике видно, как точность модели изменяется в течение обучения для каждого размера batch. Вертикальные оси - точность, горизонтальные оси - эпохи обучения. Как правило, точность должна повышаться с увеличением числа эпох, но скорость и стабильность обучения могут зависеть от размера batch
</details>