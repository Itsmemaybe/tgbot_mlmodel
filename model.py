# Импорт необходимых библиотек
import os  # Для работы с операционной системой и файлами
import numpy as np  # Для числовых операций с массивами
import tensorflow as tf  # Основной фреймворк для машинного обучения
import matplotlib.pyplot as plt  # Для визуализации данных и изображений
import matplotlib.image as mpimg  # Для работы с изображениями в matplotlib
import argparse  # Для обработки аргументов командной строки
from tensorflow.keras.preprocessing import image  # Для предобработки изображений
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Для генерации augmented данных


# Основная функция программы
def main(train_dir, valid_dir, test_image_path):
    # Настройка путей к директориям с данными
    train_people_dir = os.path.join(train_dir, 'human')
    train_dolphin_dir = os.path.join(train_dir, 'tiger')
    valid_people_dir = os.path.join(valid_dir, 'human')
    valid_dolphin_dir = os.path.join(valid_dir, 'tiger')

    # Вывод статистики по данным
    print('Total training people images:', len(os.listdir(train_people_dir)))
    print('Total training tiger images:', len(os.listdir(train_dolphin_dir)))
    print('Total validation people images:', len(os.listdir(valid_people_dir)))
    print('Total validation tiger images:', len(os.listdir(valid_dolphin_dir)))

    # Настройка отображения примеров изображений
    nrows, ncols = 4, 4  # Размер сетки для отображения (4x4)
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    # Получаем имена первых 8 изображений каждого класса
    train_people_names = os.listdir(train_people_dir)[:8]
    train_dolphin_names = os.listdir(train_dolphin_dir)[:8]

    # Создаем полные пути к изображениям
    sample_people_pics = [os.path.join(train_people_dir, fname) for fname in train_people_names]
    sample_dolphin_pics = [os.path.join(train_dolphin_dir, fname) for fname in train_dolphin_names]

    # Отображаем изображения в сетке
    for i, img_path in enumerate(sample_people_pics + sample_dolphin_pics):
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')
        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

    # Генераторы данных
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        classes=['human', 'tiger'],
        target_size=(200, 200),
        batch_size=16,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        valid_dir,
        classes=['human', 'tiger'],
        target_size=(200, 200),
        batch_size=16,
        class_mode='binary',
        shuffle=False
    )

    # Модель нейронной сети
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Компиляция модели
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Вывод структуры
    model.summary()

    # Обучение
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        verbose=1
    )

    # Сохранение модели
    model.save('people_tiger_classifier.h5')

    # Классификация тестового изображения
    img = image.load_img(test_image_path, target_size=(200, 200))
    x = image.img_to_array(img)
    plt.imshow(x / 255.)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] < 0.5:
        print(f"{test_image_path} is a person")
    else:
        print(f"{test_image_path} is a tiger")


# Точка входа
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify images as people or dolphins')

    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset directory')
    parser.add_argument('--valid_dir', type=str, required=True, help='Path to validation dataset directory')
    parser.add_argument('--test_image', type=str, required=True, help='Path to image file to classify')

    args = parser.parse_args()
    main(args.train_dir, args.valid_dir, args.test_image)
