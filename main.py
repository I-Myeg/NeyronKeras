import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Загрузка и предварительная обработка датасета MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Определение архитектуры нейронной сети
model = models.Sequential([
   layers.Flatten(input_shape=(28, 28)),
   layers.Dense(128, activation='relu'),
   layers.Dropout(0.2),
   layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Обучение нейронной сети
model.fit(x_train, y_train, epochs=5)

# Оценка точности модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Точность на тестовых данных:', test_acc)

# Сохранение обученной модели
model.save('mnist_model.h5')

# Загрузка модели из файла
model = load_model('mnist_model.h5')

# Вывод структуры модели
model.summary()

# Оценка точности модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Точность на тестовых данных:', test_acc)

# Прогнозирование классов на тестовых данных
predictions = model.predict(x_test)


# Вывод случайного изображения и соответствующего предсказания
index = np.random.randint(0, len(x_test))
plt.imshow(x_test[index], cmap='gray')
plt.axis('off')
plt.title(f'Истинная метка: {y_test[index]}, Предсказанная метка: {np.argmax(predictions[index])}')
plt.show()

