import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 차트 한글 표시
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 전이학습

# 이미지 전처리 및 증강
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
# validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 64
img_width = 150
img_height = 150

train_data = train_datagen.flow_from_directory(
    './train_dataset/train',
    batch_size=batch_size,
    target_size=(img_width, img_height),
    shuffle=True,
    subset='training'
)
valid_data = train_datagen.flow_from_directory(
    './train_dataset/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=False,
    subset='validation'
)
test_data = test_datagen.flow_from_directory(
    './train_dataset/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
)

# 받아온 이미지 데이터 출력 함수
def visualize_images(images, labels):
    figure, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 14))
    classes = list(train_data.class_indices.keys())
    img_no = 0
    for i in range(3):
        for j in range(3):
            img = images[img_no]
            label_no = np.argmax(labels[img_no])

            ax[i, j].imshow(img)
            ax[i, j].set_title(classes[label_no])
            ax[i, j].set_axis_off()
            img_no += 1

# 이미지 띄우기
# images, labels = next(train_data)
# visualize_images(images, labels)

# 전이학습 실행
base = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
base.trainable = True
model = Sequential()
model.add(base)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=1, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
check_point = ModelCheckpoint(filepath='fish_class_cnn.keras', monitor='val_accuracy', verbose=1, save_best_only=True)

history = model.fit(train_data, epochs=50, validation_data=valid_data, callbacks=[early_stop, reduce_lr, check_point])

# 정확도 및 손실 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 테스트 이미지 분류
from keras.preprocessing import image
import os

test_dir = './train_dataset/test'

test_fish_classes = os.listdir(test_dir)

for fish_class in test_fish_classes:
    filenames = os.listdir(os.path.join(test_dir, fish_class))
    fig = plt.figure(figsize=(16, 10))
    rows, cols = 1, 6
    for i, fn in enumerate(filenames):
        path = os.path.join(test_dir, fish_class, fn)
        test_img = image.load_img(path, color_mode='rgb', target_size=(150, 150), interpolation='bilinear')  # color_mode 변경
        x = image.img_to_array(test_img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        classes = model.predict(x)

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
        plt.imshow(test_img)  # cmap='gray' 제거
    plt.show()
