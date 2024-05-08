import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import os
import PIL
import shutil

# 차트 한글 표시
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 기본 경로
base_dir = 'train_dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# 훈련용 이미지 파일 이름 조회
train_fish_fnames = {}
train_fish_classes = os.listdir(train_dir)
for fish_class in train_fish_classes:
    train_fish_fnames[fish_class] = os.listdir(os.path.join(train_dir, fish_class))

# 검증용 이미지 파일 이름 조회
validation_fish_fnames = {}
for fish_class in train_fish_classes:
    validation_fish_fnames[fish_class] = os.listdir(os.path.join(validation_dir, fish_class))

# 테스트용 이미지 파일 이름 조회
test_fish_fnames = {}
for fish_class in train_fish_classes:
    test_fish_fnames[fish_class] = os.listdir(os.path.join(test_dir, fish_class))

# 이미지 데이터 전처리
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=25,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')
# validation 및 test 이미지는 augmentation을 적용하지 않는다;
# 모델 성능을 평가할 때에는 이미지 원본을 사용 (rescale만 진행)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow_from_directory() 메서드를 이용해서 훈련과 테스트에 사용될 이미지 데이터를 만들기
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=16,
                                                    color_mode='grayscale',
                                                    class_mode='categorical',
                                                    target_size=(150,150))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=4,
                                                              color_mode='grayscale',
                                                              class_mode='categorical',
                                                              target_size=(150,150))

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=4,
                                                  color_mode='grayscale',
                                                  class_mode='categorical',
                                                  target_size=(150,150))

# 합성곱 신경망 모델 구성하기
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(train_fish_classes), activation='softmax')  # 클래스 개수에 맞게 출력층 뉴런 수 조정
])
model.summary()

from tensorflow.keras.optimizers import RMSprop

# compile() 메서드를 이용해서 손실 함수 (loss function)와 옵티마이저 (optimizer)를 지정
model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=4,
                    epochs=100,
                    validation_steps=4,
                    verbose=2)

# 모델 성능 평가
print("===== train =====")
model.evaluate(train_generator)
print("===== validation =====")
model.evaluate(validation_generator)

# 정확도 및 손실 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 이제 테스트 이미지 분류
from keras.preprocessing import image

test_fish_classes = list(test_fish_fnames.keys())

# 테스트 이미지 분류
for fish_class, filenames in test_fish_fnames.items():
    fig = plt.figure(figsize=(16, 10))
    rows, cols = 1, 6
    for i, fn in enumerate(filenames):
        path = os.path.join(test_dir, fish_class, fn)
        test_img = image.load_img(path, color_mode='grayscale', target_size=(150, 150), interpolation='bilinear')
        x = image.img_to_array(test_img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        classes = model.predict(x)

        fig.add_subplot(rows, cols, i + 1)
        predicted_class_index = np.argmax(classes)
        predicted_class = test_fish_classes[predicted_class_index]

        plt.title(fn + " is " + predicted_class)
        plt.axis('off')
        plt.imshow(test_img, cmap='gray')
    plt.show()

# 모델 성능 평가
print("===== 모델 성능 평가 =====")
model.evaluate(test_generator)

# 모델 저장
# model.save('fish_class_cnn.h5')
