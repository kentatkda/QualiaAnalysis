import numpy as np
import matplotlib.pyplot as plt
from src.GWOptimalTransfer import GWOptimalTransfer
from src.CalculateDissimilarity import CalculateDissimilarity
from sklearn.cluster import KMeans

class GenerateRandomBenchmark:

    def __init__(self, qualia_color) -> None:
        self.qualia_color = qualia_color

    # 0 ~ 8 (4を除く) の間の整数でランダムな値を生成する関数
    def generate_random_value(self):
        value = np.random.randint(0, 9)
        while value == 4:
            value = np.random.randint(0, 9)
        return value
    
    def generate_random_matrix(self, size = 12):
        # 12x12の行列を作成
        random_matrix = np.zeros((size, size), dtype=int)

        # 対角成分を8に設定
        np.fill_diagonal(random_matrix, 0)

        # 下三角行列のランダムな値を設定
        for i in range(size):
            for j in range(i):
                random_value = self.generate_random_value()
                random_matrix[i, j] = random_value
                random_matrix[j, i] = random_value  # 対称行列にするため
        return random_matrix
    
    def random_transfer(self, iter_num, plot_dim, title):

        #generate random dissimilarity matix
        random_matrices = [self.generate_random_matrix() for _ in range(2)]
        emotions = list(self.qualia_color.keys())
        cal_dissim = CalculateDissimilarity(emotions=emotions)
        random_embeddings = [cal_dissim.original_embedding(dissimilarity_matrix=matrix, plot_dim=3) for matrix in random_matrices]

        colors = list(self.qualia_color.values())
        gwot = GWOptimalTransfer(emotions=emotions)

        #totally same as the emotionanalysis class
        optimal_P, optimal_Q, mapped_random_embeddings = gwot.run_personal_gwot(
            X_dissimilarity_matrix=random_matrices[0],
            Y_dissimilarity_matrix=random_matrices[1],
            X_embedding=random_embeddings[0],
            Y_embedding=random_embeddings[1],
            iter_num=iter_num
        )
        print(f'mapped mean embeddings:{np.array(mapped_random_embeddings).shape}')

        markers = ['o', 'x']
        labels=['non_al(group1)', 'non_al(group2)']
        n_clusters = 3
        handles = []

        fig = plt.figure(figsize=(12,9))

        if plot_dim==2:
            ax = fig.add_subplot(2,2,1)
            for i in range(len(mapped_random_embeddings)):
                for j in range(len(mapped_random_embeddings[0])):
                    scatter = ax.scatter(mapped_random_embeddings[i][j,0], mapped_random_embeddings[i][j,1], color=colors[j], marker=markers[i], label=labels[i])
                    if j==0:
                        handles.append(scatter)

        elif plot_dim==3:
            ax = fig.add_subplot(2,2,1, projection='3d')
            for i in range(len(mapped_random_embeddings)):
                for j in range(len(mapped_random_embeddings[0])):
                    scatter = ax.scatter(mapped_random_embeddings[i][j,0], mapped_random_embeddings[i][j,1], mapped_random_embeddings[i][j,2], color=colors[j], marker=markers[i], label=labels[i])
                    if j==0:
                        handles.append(scatter)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if plot_dim==3:
            ax.set_zlabel('Z')
        ax.set_title(f'{title}')
        ax.grid(True)

        # 凡例を追加
        ax.legend(handles, labels)

        # 各感情ごとにデータを収集
        emotion_embeddings = []
        for i in range(12):
            emotion_embeddings.append(np.vstack([mapped_random_embeddings[j][i] for j in range(len(mapped_random_embeddings))]))

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
        handles = dict()
        #プロット
        if plot_dim ==2:
            ax = fig.add_subplot(2,2,2)
            handle_cluster_flags = np.zeros(n_clusters)
            # 各感情のデータポイントをクラスタごとにプロット
            for i, embedding in enumerate(emotion_embeddings):
                for j in range(len(mapped_random_embeddings)):
                    scatter = ax.scatter(embedding[j, 0], embedding[j, 1],  color=emotion_colors[i], label=f'Cluster {emotion_labels[i]}' if j == 0 else "")
                    if handle_cluster_flags[emotion_labels[i]] == 0:
                        handles[f'Cluster {emotion_labels[i]}'] = scatter
                        handle_cluster_flags[emotion_labels[i]] = 1
                    ax.scatter(embedding[j, 0], embedding[j, 1], edgecolor=emotion_colors[i], facecolor='none')
        elif plot_dim==3:
            ax = fig.add_subplot(2,2,2, projection='3d')
            handle_cluster_flags = np.zeros(n_clusters)
            # 各感情のデータポイントをクラスタごとにプロット
            for i, embedding in enumerate(emotion_embeddings):
                for j in range(len(mapped_random_embeddings)):
                    scatter = ax.scatter(embedding[j, 0], embedding[j, 1], embedding[j, 2],  color=emotion_colors[i])
                    if handle_cluster_flags[emotion_labels[i]] == 0:
                        handles[f'Cluster {emotion_labels[i]}'] = scatter
                        handle_cluster_flags[emotion_labels[i]] = 1
                    ax.scatter(embedding[j, 0], embedding[j, 1],  embedding[j, 2], edgecolor=emotion_colors[i], facecolor='none')


        # グラフの装飾
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if plot_dim==3:
            ax.set_zlabel('Y')
        ax.set_title(f'{plot_dim}D Embeddings with {n_clusters} Clusters')
        ax.grid(True)
        ax.legend(list(handles.values()), list(handles.keys()))


        #感情が属しているクラス分類
        cluster_emotions = {cluster: [] for cluster in range(n_clusters)}
        for i, emotion in enumerate(emotions):
            cluster_emotions[emotion_labels[i]].append(emotion)
        print("\nCluster Emotions:")
        for cluster, emotions_ in cluster_emotions.items():
            print(f"Cluster {cluster}: {emotions_}")

        np.array(mapped_random_embeddings).shape
        #感情間の距離の表示
        embeddings_diff = mapped_random_embeddings[0] - mapped_random_embeddings[1]
        distances = [(diff[0]**2 + diff[1]**2)**0.5 for diff in embeddings_diff]
        # 棒グラフの作成
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(2,2,2)
        ax.bar(emotions, distances, color='skyblue')
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Distance')
        ax.set_title(f'Non Alcohol->Alcohol Mean: Distance for Each Emotion')
        ax.set_xticks(range(len(emotions)))  # ティックの位置を設定
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.grid(axis='y')
        plt.show()
        random_distances = distances
        return random_distances
