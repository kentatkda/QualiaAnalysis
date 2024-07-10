import numpy as np
import ot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
class GWOptimalTransfer:
    """
    X, Y(12*12)の非類似度行列から
    最適輸送法を用いて最適マッピングを求める。
    """

    def __init__(self,emotions):
        self.emotions = emotions


    def gw_optimization(self, D1, D2, iter_num, epsilon=0.2):
        p, q = ot.unif(D1.shape[0]), ot.unif(D2.shape[0])
        # print(f'D1, D2: {D1.shape, D2.shape}')

        best_gw_distance = float('inf')
        optimal_P = None
        for _ in tqdm(range(iter_num)):
            p = np.random.rand(len(D1))
            p /= p.sum()
            q = np.random.rand(len(D2))
            q /= q.sum()
            T, log = ot.gromov.entropic_gromov_wasserstein(D1, D2, p, q, log=True, solver='PGD', epsilon=epsilon)
            if log['gw_dist'] < best_gw_distance:
                best_gw_distance = log['gw_dist']
                optimal_P = T
        # print(f'optimal_P: {optimal_P.shape}')
        print(f'best_gw_distance: {best_gw_distance}')

        return optimal_P, best_gw_distance
    
    def calcurate_optimal_Q(
            self,
            X_embedding,
            Y_embedding,
            optimal_P
        ):
        XYP_T = np.dot(X_embedding, np.dot(Y_embedding, optimal_P).T)

        # 特異値分解
        U, S, Vt = np.linalg.svd(XYP_T)

        # Q* = UV^T の計算
        optimal_Q = np.dot(U, Vt)
        return optimal_Q
    
    def plot_optimal_P_heatmap(self, optimal_P):
        fig = plt.figure(figsize=(12, 8))
        #optimal_Pの可視化part2
        ax = fig.add_subplot(1, 1, 1)
        sns.heatmap(optimal_P, cmap='Reds')
        ax.set_title('Y→X heatmap of optimal P matrix')
        ax.set_xlabel('Y Indices')
        ax.set_ylabel('X Indices')

    def run_personal_gwot(
        self,
        X_dissimilarity_matrix,
        Y_dissimilarity_matrix,
        X_embedding,
        Y_embedding,
        iter_num
    ):
        D1 = Y_dissimilarity_matrix
        D2 = X_dissimilarity_matrix
        X_embedding = X_embedding.T
        Y_embedding = Y_embedding.T

        optimal_P, best_gw_distance = self.gw_optimization(D1, D2, iter_num)

        optimal_Q = self.calcurate_optimal_Q(
            X_embedding=X_embedding,
            Y_embedding=Y_embedding,
            optimal_P=optimal_P)

        optimal_transfered_Y = np.dot(optimal_Q, Y_embedding)
        X = X_embedding.T
        Y = optimal_transfered_Y.T
        personal_embeddings = [X,Y]

        return optimal_P, optimal_Q, personal_embeddings
    
    def plot_personal_embedding_and_distance(
            self,
            personal_embeddings,
            subject,
            colors
    ):
        """
        得られた１被験者に対する非飲酒時ベクトルと飲酒時の最適輸送後ベクトル
        から、それらの同時プロットグラフと感情間の距離を感情ごとにプロットしたグラフを出力
        """
        labels = ['X', 'Y']
        markers = ['o', 'x']
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(2,2,1)
        for j in range(len(personal_embeddings)):
            for i in range(12):
                ax.scatter(personal_embeddings[j][i, 0], personal_embeddings[j][i, 1], color=colors[i], marker=markers[j], label=labels[j])

        # 凡例の作成
        handles = [plt.Line2D([0], [0], marker=markers[j], color='w', markerfacecolor='k', markersize=10, linestyle='None', label=labels[j]) for j in range(len(personal_embeddings))]

        # グラフの装飾
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'subject:{subject} X & QY embeddings')
        ax.legend(handles=handles)


        #感情間の距離の表示
        embeddings_diff = personal_embeddings[0] - personal_embeddings[1]
        distances = [(diff[0]**2 + diff[1]**2)**0.5 for diff in embeddings_diff]
        # 棒グラフの作成
        ax = fig.add_subplot(2,2,2)
        ax.bar(self.emotions, distances, color='skyblue')
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Distance')
        ax.set_title(f'{subject}: Distance for Each Emotion')
        ax.set_xticks(range(len(self.emotions)))  # ティックの位置を設定
        ax.set_xticklabels(self.emotions, rotation=45, ha='right')
        ax.grid(axis='y')


        # グラフの表示
        plt.tight_layout()
        plt.show()

        return distances

