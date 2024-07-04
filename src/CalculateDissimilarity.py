import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class CalculateDissimilarity:
    """
    前処理されたデータを基に非類似行列を生成、それを基に埋め込みベクトルを生成しプロットするクラス
    """
    def __init__(self, emotions):
        self.emotions = emotions

    def dissimilarity_matrix(self, preprocess_df:pd.DataFrame):
        # 12x12の行列を初期化
        dissimilarity_matrix = np.zeros((12, 12))
        # カテゴリからインデックスを作成
        emotion_index = {emotion: idx for idx, emotion in enumerate(self.emotions)}

        # 非類似度行列に変換
        # max_similarity = np.max(df['similarity'])
        max_similarity=4
        # データフレームの内容を行列に反映
        for _, row in preprocess_df.iterrows():
            emotion1, emotion2 = row['compared_video']
            similarity = row['similarity']
            idx1 = emotion_index[emotion1]
            idx2 = emotion_index[emotion2]
            #距離に反映させるために非類似度に変換
            dissimilarity_matrix[idx1, idx2] = max_similarity - similarity
            dissimilarity_matrix[idx2, idx1] = max_similarity - similarity  # 対称行列にする
        return dissimilarity_matrix


    def original_embedding(self, dissimilarity_matrix, plot_dim):
        # 類似度行列は対称行列にする(既に対称なはず)
        matrix = dissimilarity_matrix
        matrix = (matrix + matrix.T) / 2

        # Multidimensional Scaling (MDS) を使用して3次元空間に埋め込む
        mds = MDS(n_components=plot_dim, dissimilarity='precomputed', random_state=42)
        original_embedding = mds.fit_transform(matrix)
        return original_embedding
