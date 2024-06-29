import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CalculateDissimilarity:
  """
  前処理されたデータを基に類似行列を生成、それを基に埋め込みベクトルを生成しプロットするクラス
  """
  def __init__(self, preprocessed_dfs:pd.DataFrame, qualia_color:dict):
    self.preprocessed_dfs = preprocessed_dfs
    self.qualia_color = qualia_color
    self.emotions = list(self.qualia_color.keys())

  def dissimilarity_matrices(self):
    dissimilarity_matrices = list()
    for df in self.preprocessed_dfs:
      # 12x12の行列を初期化
      dissimilarity_matrix = np.zeros((12, 12))
      # カテゴリからインデックスを作成
      emotion_index = {emotion: idx for idx, emotion in enumerate(self.emotions)}

      # 非類似度行列に変換
      max_similarity = np.max(df['similarity'])
      # データフレームの内容を行列に反映
      for _, row in df.iterrows():
        emotion1, emotion2 = row['compared_video']
        similarity = row['similarity']
        idx1 = emotion_index[emotion1]
        idx2 = emotion_index[emotion2]
        #距離に反映させるために非類似度に変換
        dissimilarity_matrix[idx1, idx2] = max_similarity - similarity
        dissimilarity_matrix[idx2, idx1] = max_similarity - similarity  # 対称行列にする
      dissimilarity_matrices.append(dissimilarity_matrix)

    return dissimilarity_matrices

  def dissimilarity_dfs(self, dissimilarity_matrices:list):
    dissimilarity_dfs = [pd.DataFrame(matrix, index=self.emotions, columns=self.emotions) for matrix in dissimilarity_matrices]
    return dissimilarity_dfs

  def original_embeddings(self, dissimilarity_matrices:list , plot_dim:int):
    original_embeddings_list = list()
    for matrix in dissimilarity_matrices:
      # 類似度行列は対称行列にする(既に対称なはず)
      matrix = (matrix + matrix.T) / 2

      # Multidimensional Scaling (MDS) を使用して3次元空間に埋め込む
      mds = MDS(n_components=plot_dim, dissimilarity='precomputed', random_state=42)
      original_embedding = mds.fit_transform(matrix)

      #save
      original_embeddings_list.append(original_embedding)
    return original_embeddings_list

  def plot_in_space(self, all_original_embeddings_list:list , attributes:list,  plot_dim:int=3):

    if plot_dim == 3:
      fig = plt.figure(figsize=(12, 6))
      for i, original_embeddings_list in enumerate(all_original_embeddings_list):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        for embedding in original_embeddings_list:
          embedding_df = pd.DataFrame(embedding, index=self.emotions, columns=['Dim1', 'Dim2', 'Dim3'])
          colors = [self.qualia_color[emotion] for emotion in embedding_df.index]

          ax.scatter(embedding_df['Dim1'], embedding_df['Dim2'], embedding_df['Dim3'], c=colors)

        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')
        ax.set_zlabel('Dim3')

        ax.set_title(f'{attributes[i]} Embedding of Emotional Categories in 3D Space')

      plt.suptitle(f'Non_Al / Al Embedding of Emotional Categories')
      plt.tight_layout()
      plt.show()
    elif plot_dim == 2:
      fig = plt.figure(figsize=(10, 8))
      for i, original_embeddings_list in enumerate(all_original_embeddings_list):
        ax = fig.add_subplot(2, 2, i+1)
        for embedding in original_embeddings_list:
          embedding_df = pd.DataFrame(embedding, index=self.emotions, columns=['Dim1', 'Dim2'])
          colors = [self.qualia_color[emotion] for emotion in embedding_df.index]

          ax.scatter(embedding_df['Dim1'], embedding_df['Dim2'], c=colors)

        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')

        ax.set_title(f'{attributes[i]} Embedding of Emotional Categories')

      plt.suptitle(f'Non_Al / Al Embedding of Emotional Categories in {plot_dim}D Space')
      plt.tight_layout()
      plt.show()