import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Cluster:
    """
    X, QiYi(i=1~n)のような
    元の埋め込みベクトルと最適マップ後のベクトルを同時にプロットしたようなグラフにおいて
    指定した数のクラスにクラスタリングするためのクラス
    """

    def __init__(self, all_embeddings, qualia_color:dict):
        #self.all_embeddings = (9, 12, 2) or (9, 12, 3)
        self.all_embeddings = np.array(list(all_embeddings.values()))
        self.data_n = len(self.all_embeddings)
        self.qualia_color = qualia_color

        self.colors = list(self.qualia_color.values())
        self.emotions = list(self.qualia_color.keys())


    def plot_unclustered_embeddings(self, fig):
        labels = ['X'] + [f'Y{i}' for i in range(1,self.data_n)]
        markers = ['o', 'x', 's', 'd', '^', 'v', '>', '<', '*']

        ax = fig.add_subplot(2,2,1)
        for j in range(len(self.all_embeddings)):
            for i in range(self.all_embeddings.shape[1]):
                ax.scatter(self.all_embeddings[j][i, 0], self.all_embeddings[j][i, 1], color=self.colors[i], marker=markers[j], label=labels[j])

        # 凡例の作成
        handles = [plt.Line2D([0], [0], marker=markers[j], color='w', markerfacecolor='k', markersize=10, linestyle='None', label=labels[j]) for j in range(len(self.all_embeddings))]

        # グラフの装飾
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('X_embeddings and QiYi(i=1~8) embeddings')
        ax.legend(handles=handles)

        return fig

    def plot_clustered_embeddings(self, fig, n_clusters):
        # 各感情ごとにデータを収集
        emotion_embeddings = []
        for i in range(12):
            emotion_embeddings.append(np.vstack([self.all_embeddings[j][i] for j in range(len(self.all_embeddings))]))

        emotion_embeddings = np.array(emotion_embeddings)  # (12, 9, 2)の形状に変換

        # 各感情ごとにデータをフラット化
        flattened_embeddings = emotion_embeddings.reshape(12, -1)  # (12, 10)の形状に変換

        # k-meansクラスタリングの適用
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_embeddings)

        # クラスタのラベル取得
        emotion_labels = kmeans.labels_

        # クラスタごとの色設定
        cluster_colors = ['r', 'g', 'b', 'y', 'c']
        emotion_colors = [cluster_colors[label % len(cluster_colors)] for label in emotion_labels]

        #プロット
        ax = fig.add_subplot(2,2,2)

        # 各感情のデータポイントをクラスタごとにプロット
        for i, embedding in enumerate(emotion_embeddings):
            for j in range(len(self.all_embeddings)):
                ax.scatter(embedding[j, 0], embedding[j, 1], color=emotion_colors[i], label=f'Cluster {emotion_labels[i]}' if j == 0 else "")
                ax.scatter(embedding[j, 0], embedding[j, 1], edgecolor=emotion_colors[i], facecolor='none')

        # クラスタの凡例の追加
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[i], markersize=10, linestyle='None', label=f'Cluster {i}') for i in range(n_clusters)]
        ax.legend(handles=handles, title="Clusters")

        # グラフの装飾
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('2D Scatter Plot of Embeddings with Matching Clusters')
        ax.grid(True)

        #感情が属しているクラス分類
        cluster_emotions = {cluster: [] for cluster in range(n_clusters)}
        for i, emotion in enumerate(self.emotions):
            cluster_emotions[emotion_labels[i]].append(emotion)
        print("\nCluster Emotions:")
        for cluster, emotions in cluster_emotions.items():
            print(f"Cluster {cluster}: {emotions}")

        return fig, cluster_emotions

    def plot_all(self, n_clusters):
        fig = plt.figure(figsize=(12,9))
        fig1 = self.plot_unclustered_embeddings(fig=fig)
        fig2, cluster_emotions = self.plot_clustered_embeddings(fig=fig1, n_clusters=n_clusters)
        plt.show()
        return cluster_emotions