import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.api import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image

# 차트 한글 표시
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# DNN모델

# 기본 경로
base_dir = 'train_dataset'
train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# 훈련용 이미지 파일 이름 조회
train_fish_fnames = {}
train_fish_classes = os.listdir(train_dir)
for fish_class in train_fish_classes:
    train_fish_fnames[fish_class] = os.listdir(os.path.join(train_dir, fish_class))

# 검증용 이미지 파일 이름 조회
# validation_fish_fnames = {}
# for fish_class in train_fish_classes:
#     validation_fish_fnames[fish_class] = os.listdir(os.path.join(validation_dir, fish_class))

# 테스트용 이미지 파일 이름 조회
test_fish_fnames = {}
for fish_class in train_fish_classes:
    test_fish_fnames[fish_class] = os.listdir(os.path.join(test_dir, fish_class))

# 이미지 데이터 전처리
# Image augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,  # 이미지를 최대 45도까지 회전
    width_shift_range=0.2,  # 최대 20%의 너비 이동
    height_shift_range=0.2,  # 최대 20%의 높이 이동
    shear_range=0.2,  # 최대 20%의 전단 변형
    zoom_range=0.2,  # 최대 20%의 확대/축소
    horizontal_flip=True,  # 수평 뒤집기
    vertical_flip=True,  # 수직 뒤집기
    fill_mode='nearest',  # 이미지 변환 후 생기는 빈 공간을 채우는 방법
    brightness_range=[0.5, 1.5],
    validation_split=0.2
    # validation_set으로 쓸 데이터의 비율을 정한다.
    # 만약 값이 정해져 있다면 후에 사용할 flow_from_directory나 flow_from_dataframe에서 파라미터 subset = 'training' 혹은 subset = 'validation'으로
    # 훈련 데이터 셋과 검증 데이터 셋을 줄 수 있다.
)
# validation 및 test 이미지는 augmentation을 적용하지 않는다;
# 모델 성능을 평가할 때에는 이미지 원본을 사용 (rescale만 진행)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# flow_from_directory() 메서드를 이용해서 훈련과 테스트에 사용될 이미지 데이터를 만들기
# 이미지 데이터를 로드하는 데이터 제너레이터 생성
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=128,
                                                    color_mode='rgb',
                                                    class_mode='categorical',
                                                    target_size=(150, 150),
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory(train_dir,
                                                         batch_size=64,
                                                         color_mode='rgb',
                                                         class_mode='categorical',
                                                         target_size=(150, 150),
                                                         subset='validation')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=64,
                                                  color_mode='rgb',
                                                  class_mode='categorical',
                                                  target_size=(150, 150))


# DNN 모델 구성
dnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(150, 150, 3)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(train_fish_classes), activation='softmax')
])
dnn_model.summary()

# 컴파일
dnn_model.compile(optimizer=RMSprop(learning_rate=0.0005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# 훈련
dnn_history = dnn_model.fit(train_generator,
                             validation_data=validation_generator,
                             steps_per_epoch=4,
                             epochs=100,
                             validation_steps=4,
                             verbose=2)

# 성능 평가
print("===== train =====")
dnn_model.evaluate(train_generator)
print("===== validation =====")
dnn_model.evaluate(validation_generator)

# 정확도 및 손실 시각화
dnn_acc = dnn_history.history['accuracy']
dnn_val_acc = dnn_history.history['val_accuracy']
dnn_loss = dnn_history.history['loss']
dnn_val_loss = dnn_history.history['val_loss']

epochs = range(len(dnn_acc))

plt.plot(epochs, dnn_acc, label='DNN Training accuracy')
plt.plot(epochs, dnn_val_acc, label='DNN Validation accuracy')
plt.title('DNN Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, dnn_loss, label='DNN Training loss')
plt.plot(epochs, dnn_val_loss, label='DNN Validation loss')
plt.title('DNN Training and validation loss')
plt.legend()

plt.show()

# 이제 테스트 이미지 분류

test_fish_classes = list(test_fish_fnames.keys())

# 테스트 이미지 분류
for fish_class, filenames in test_fish_fnames.items():
    fig = plt.figure(figsize=(12, 8))
    rows, cols = 1, 6
    for i, fn in enumerate(filenames):
        path = os.path.join(test_dir, fish_class, fn)
        test_img = image.load_img(path, color_mode='rgb', target_size=(150, 150), interpolation='bilinear')
        x = image.img_to_array(test_img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        classes = dnn_model.predict(x)

        # 가장 높은 확률을 가진 클래스의 인덱스와 확률을 가져옴
        top_class_index = np.argmax(classes)
        top_class_prob = np.max(classes)

        # 예측된 클래스의 이름 가져오기
        predicted_class = test_fish_classes[top_class_index]

        # 클래스별 확률을 표시할 문자열 생성
        class_probs = "\n".join([f"{test_fish_classes[i]}: {prob * 100:.2f}%" for i, prob in enumerate(classes[0])])

        # 이미지와 분류 결과 및 확률을 표시
        fig.add_subplot(rows, cols, i + 1)
        plt.title(f"{fn} is {predicted_class}\n\n{class_probs}")
        plt.axis('off')
        plt.imshow(test_img, cmap='gray')
    plt.show()

# 모델 성능 평가
print("===== 모델 성능 평가 =====")
dnn_model.evaluate(test_generator)

# 모델 저장
print("===== 모델 저장 =====")
dnn_model.save('fish_class_dnn.keras')
