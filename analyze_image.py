import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

image_path='train_dataset/test/림포시스티스병/림포시스티스병 넙치 4.jpg'
# 이미지를 분석하는 함수
def analyze_image(image_path):
    # 각 모델의 경로 설정
    model_paths = ['fish_class_cnn.keras', 'fish_class_tl.keras', 'fish_class_dnn.keras']
    # 결과를 저장할 리스트
    results = []

    # 각 모델에 대해 반복
    for idx, model_path in enumerate(model_paths):
        # TensorFlow 모델 로드
        model = tf.keras.models.load_model(model_path)

        # 이미지 불러오기 및 전처리
        img = image.load_img(image_path, target_size=(150, 150))  # 이미지 크기를 모델에 맞게 조정
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # 예측 수행
        prediction = model.predict(img_array)

        # 클래스 레이블 가져오기
        classes = ['림포시스티스병', '비브리오', '아가미흡충', '연쇄구균병']  # 클래스 레이블을 적절하게 수정

        # 가장 높은 확률을 가진 클래스의 인덱스와 확률을 가져오기
        top_class_index = np.argmax(prediction)
        top_class_prob = np.max(prediction)

        # 예측된 클래스와 해당 클래스의 확률을 튜플로 저장하여 리스트에 추가
        result = (f"{['CNN', '전이학습', 'DNN'][idx]} 모델", classes[top_class_index], top_class_prob)
        results.append(result)

    return results


if __name__ == "__main__":
    # 이미지 분석 결과 출력
    result = analyze_image(image_path)
    print(result)
