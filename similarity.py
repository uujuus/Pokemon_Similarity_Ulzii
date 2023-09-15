import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model  # Model을 추가합니다.
from tensorflow.keras.applications import vgg16
import pandas as pd

class Image_Similarity:
    def __init__(self, folder_path):
        self.folder_path = folder_path  # 이미지 폴더 경로를 클래스 속성으로 저장
        self.image_extensions = ['.jpg', '.jpeg', '.png']
        self.image_files = self.get_image_files()

        self.extract_model = self.load_vgg_model()

    def get_image_files(self):
        image_files = []
        for file_name in os.listdir(self.folder_path):
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension in self.image_extensions:
                image_files.append(os.path.join(self.folder_path, file_name))
        return image_files

    def load_vgg_model(self):
        vgg_model = vgg16.VGG16(weights='imagenet')
        extract_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc1").output)
        return extract_model

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB")
        x = np.array(image)
        x = np.expand_dims(x, axis=0)
        x = vgg16.preprocess_input(x)
        return x

    def extract_features(self, image_paths):
        feature_vectors = []
        for image_path in image_paths:
            x = self.preprocess_image(image_path)
            feature = self.extract_model.predict(x)[0]
            feature_vectors.append(feature)
        return np.array(feature_vectors)

    def calculate_cosine_similarity(self, feature_vectors):
        return cosine_similarity(feature_vectors)

    def save_similarity_matrix_to_csv(self, similarity_matrix, output_file):
        # 데이터프레임으로 변환하고 소수점 둘째 자리까지 반올림
        similarity_df = pd.DataFrame(similarity_matrix, columns=self.image_files, index=self.image_files)
        similarity_df = similarity_df.round(2)
        
        # CSV 파일로 저장
        similarity_df.to_csv(output_file)

