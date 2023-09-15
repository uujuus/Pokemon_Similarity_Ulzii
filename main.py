from similarity import Image_Similarity
import os

if __name__ == "__main__":
    folder_path = "./pokemon/images"

    calculator = Image_Similarity(folder_path)  # folder_path를 전달하여 클래스를 인스턴스화합니다
    feature_vectors = calculator.extract_features(calculator.image_files)  # image_files를 전달하여 feature_vectors를 추출합니다
    cosine_similarities = calculator.calculate_cosine_similarity(feature_vectors)  # 유사도를 계산합니다

    # 유사도 결과를 소수점 둘째 자리까지 반올림하여 CSV 파일로 저장
    calculator.save_similarity_matrix_to_csv(cosine_similarities, 'pokemon_similarity_rounded.csv')

