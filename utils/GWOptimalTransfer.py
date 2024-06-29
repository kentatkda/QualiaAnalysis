import numpy as np
import ot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class GWOptimalTransfer:

    def __init__(self, X_embeddings, Y_embeddings, qualia_color):
        self.X_embeddings = np.array(X_embeddings).copy()
        self.Y_embeddings = np.array(Y_embeddings).copy()
        self.qualia_color = qualia_color

    def reshape_embeddings(self):
        #size=3*180(2*180)に変換
        self.sample_size, self.n, self.plot_dim = self.X_embeddings.shape
        X_embeddings_combined = self.X_embeddings.transpose(1, 0, 2).reshape(self.n*self.sample_size, self.plot_dim).T
        Y_embeddings_combined = self.Y_embeddings.transpose(1, 0, 2).reshape(self.n*self.sample_size, self.plot_dim).T
        return X_embeddings_combined, Y_embeddings_combined

    def pairwise_distances(self, X, Y):
        return np.linalg.norm(X[:, :, np.newaxis] - Y[:, np.newaxis, :], axis=0)

    def gromov_wasserstein(self, X_embeddings_combined, Y_embeddings_combined, max_iter):
        C1 = self.pairwise_distances(X_embeddings_combined, X_embeddings_combined)
        C2 = self.pairwise_distances(Y_embeddings_combined, Y_embeddings_combined)
        # 初期分布（均等な分布を仮定）
        p = np.ones((self.sample_size*self.n,)) / (self.sample_size*self.n)
        q = np.ones((self.sample_size*self.n,)) / (self.sample_size*self.n)

        # Gromov-Wasserstein距離の計算
        optimal_P, log = ot.gromov.gromov_wasserstein(C2, C1, p, q, 'square_loss', log=True, max_iter=max_iter)
        return optimal_P
    
    def plot_optimal_P_heatmap(self, optimal_P):
        fig = plt.figure(figsize=(12, 8))

        #optimal_Pの可視化part2
        ax = fig.add_subplot(2, 2, 2)
        sns.heatmap(optimal_P, cmap='Reds')
        ax.set_title('Y→X heatmap of optimal P matrix')
        ax.set_xlabel('Y Indices')
        ax.set_ylabel('X Indices')

    def calcurate_Q(self, X_embeddings_combined, Y_embeddings_combined, optimal_P):
        #特異値分解によるQ行列の計算
        # (X Y P^*)^T の計算
        XYP_T = np.dot(X_embeddings_combined, np.dot(Y_embeddings_combined, optimal_P).T)

        # 特異値分解
        U, S, Vt = np.linalg.svd(XYP_T)

        # Q* = UV^T の計算
        optimal_Q = np.dot(U, Vt)

        return optimal_Q
    
    def calcurate_mapped_Y(self, optimal_Q, optimal_P, Y_embeddings_combined):
        # Y_optimal_mapped = np.dot(optimal_Q, np.dot(Y_embeddings_combined, optimal_P))
        Y_optimal_mapped = np.dot(optimal_Q, Y_embeddings_combined)
        return Y_optimal_mapped
    
    def plot_optimal(self, X_embeddings_combined, Y_optimal_mapped):
        colors = list(self.qualia_color.values())
        if self.plot_dim == 3:
            # 3次元プロットの準備
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # 座標をプロット
            colors_combined = colors * self.sample_size
            ax.scatter(Y_optimal_mapped[0], Y_optimal_mapped[1], Y_optimal_mapped[2], c=colors_combined, marker='x', label='optimal_mapped_Y')

            #元埋め込みも同様にプロット
            colors_combined = colors * self.sample_size
            embedding_df = pd.DataFrame(X_embeddings_combined.T, index=colors_combined, columns=['Dim1', 'Dim2', 'Dim3'])
            ax.scatter(embedding_df['Dim1'], embedding_df['Dim2'], embedding_df['Dim3'], c=colors_combined, marker='o', label='raw_X')


            # 軸ラベルの設定
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # グラフ表示
            plt.legend()
            plt.show()

        elif self.plot_dim == 2:

            # 2次元プロットの準備
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(1, 1, 1)

            # 座標をプロット
            colors_combined = colors * self.sample_size
            ax.scatter(Y_optimal_mapped[0], Y_optimal_mapped[1], c=colors_combined, marker='x',s=15, label='optimal_mapped_Y')

            #元埋め込みも同様にプロット
            colors_combined = colors * self.sample_size
            embedding_df = pd.DataFrame(X_embeddings_combined.T, index=colors_combined, columns=['Dim1', 'Dim2'])
            ax.scatter(embedding_df['Dim1'], embedding_df['Dim2'], c=colors_combined, marker='o',s=15, label='raw_X')


            # 軸ラベルの設定
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # グラフ表示
            plt.legend()
            plt.show()
