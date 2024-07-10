import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from runs.run1 import run1_preprocess, run1_calculate_dissimilarity
from src.extract_paths import extract_paths_from_2paths
from src.GWOptimalTransfer import GWOptimalTransfer
from src.CalculateDissimilarity import CalculateDissimilarity
from sklearn.cluster import KMeans


class EmotionAnalysis:
    def __init__(self, qualia_color):
        self.qualia_color = qualia_color
        self.emotions = list(self.qualia_color.keys())
        self.colors = list(self.qualia_color.values())

    def process1(self, non_al_folder_path, al_folder_path, plot_dim, iter_num, title, labels):

        #init
        emotions = list(self.qualia_color.keys())
        colors = list(self.qualia_color.values())

        first_paths, second_paths = extract_paths_from_2paths(
            first_folder_path=non_al_folder_path,
            second_folder_path=al_folder_path
        )
        # print(f'first_paths:{len(first_paths)}, second_paths: {len(second_paths)}')
        subjects, all_preprocessed_dfs = run1_preprocess(
            first_paths=first_paths,
            second_paths=second_paths
        )
        all_dissimilarity_matrices, all_original_embeddings = run1_calculate_dissimilarity(
                all_preprocessed_dfs=all_preprocessed_dfs,
                emotions=emotions,
                plot_dim=plot_dim,
            )
        print(f'all_dissimilarity_matrices:{all_dissimilarity_matrices.shape}')
        print(f'all_original_embeddings: {all_original_embeddings.shape}')


        cal_dissim = CalculateDissimilarity(emotions=emotions)
        gwot = GWOptimalTransfer(emotions=emotions)
        mean_matrices = list()
        mean_embeddings = list()


        for matrices in all_dissimilarity_matrices:
            mean_matrix = sum(matrices) / len(matrices)
            mean_matrices.append(mean_matrix)
            mean_embedding = cal_dissim.original_embedding(
                dissimilarity_matrix=mean_matrix,
                plot_dim=plot_dim
            )
            mean_embeddings.append(mean_embedding)

        print(np.array(mean_embeddings).shape)

        optimal_P, optimal_Q, mapped_mean_embeddings = gwot.run_personal_gwot(
            X_dissimilarity_matrix=mean_matrices[0],
            Y_dissimilarity_matrix=mean_matrices[1],
            X_embedding=mean_embeddings[0],
            Y_embedding=mean_embeddings[1],
            iter_num=iter_num
        )
        print(f'mapped mean embeddings:{np.array(mapped_mean_embeddings).shape}')

        markers = ['o', 'x']
        # labels=['non_al(group1)', 'non_al(group2)']
        n_clusters = 3
        handles = []

        fig = plt.figure(figsize=(12,9))

        if plot_dim==2:
            ax = fig.add_subplot(2,2,1)
            for i in range(len(mapped_mean_embeddings)):
                for j in range(len(mapped_mean_embeddings[0])):
                    scatter = ax.scatter(mapped_mean_embeddings[i][j,0], mapped_mean_embeddings[i][j,1], color=colors[j], marker=markers[i], label=labels[i])
                    if j==0:
                        handles.append(scatter)

        elif plot_dim==3:
            ax = fig.add_subplot(2,2,1, projection='3d')
            for i in range(len(mapped_mean_embeddings)):
                for j in range(len(mapped_mean_embeddings[0])):
                    scatter = ax.scatter(mapped_mean_embeddings[i][j,0], mapped_mean_embeddings[i][j,1], mapped_mean_embeddings[i][j,2], color=colors[j], marker=markers[i], label=labels[i])
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
            emotion_embeddings.append(np.vstack([mapped_mean_embeddings[j][i] for j in range(len(mapped_mean_embeddings))]))

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
                for j in range(len(mapped_mean_embeddings)):
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
                for j in range(len(mapped_mean_embeddings)):
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

        np.array(mapped_mean_embeddings).shape
        #感情間の距離の表示
        embeddings_diff = mapped_mean_embeddings[0] - mapped_mean_embeddings[1]
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
        mean_distances_nonal_al = distances
        return mean_distances_nonal_al
    def process2(self, non_al_folder_path, plot_dim, iter_num):
        paths = extract_paths(folder_path=non_al_folder_path)
        print(f'paths{len(paths)}')
        subjects, preprocessed_dfs = run_preprocess(paths=paths)
        dissimilarity_matrices, original_embeddings = run_calculate_dissimilarity(
            preprocessed_dfs=preprocessed_dfs,
            emotions=self.emotions,
            plot_dim=3
        )
        print(f'dissimilarity_matrices:{dissimilarity_matrices.shape}')
        print(f'original_embeddings: {original_embeddings.shape}')

        s = len(dissimilarity_matrices)//2
        group1_matrices = dissimilarity_matrices[:s]
        group2_matrices = dissimilarity_matrices[s:2*s]
        print(f'group1_matrices: {group1_matrices.shape}')

        mean_matrices = list()
        mean_embeddings = list()
        cal_dissim = CalculateDissimilarity(emotions=self.emotions)
        for matrices in [group1_matrices, group2_matrices]:
            mean_matrix = sum(matrices) / len(matrices)
            mean_matrices.append(mean_matrix)
            mean_embedding = cal_dissim.original_embedding(
                dissimilarity_matrix=mean_matrix,
                plot_dim=plot_dim
            )
            mean_embeddings.append(mean_embedding)
        print(f'mean_matrices: {np.array(mean_matrices).shape}')
        print(f'mean_embeddings: {np.array(mean_embeddings).shape}')

        gwot = GWOptimalTransfer(emotions=self.emotions)
        optimal_P, optimal_Q, mapped_mean_embeddings = gwot.run_personal_gwot(
            X_dissimilarity_matrix=mean_matrices[0],
            Y_dissimilarity_matrix=mean_matrices[1],
            X_embedding=mean_embeddings[0],
            Y_embedding=mean_embeddings[1],
            iter_num=iter_num
        )
        print(f'mapped mean embeddings:{np.array(mapped_mean_embeddings).shape}')

        markers = ['o', 'x']
        labels=['non_al', 'al']
        n_clusters = 3
        handles = []

        fig = plt.figure(figsize=(12,9))
        if plot_dim==2:
            ax = fig.add_subplot(2,2,1)
            for i in range(len(mapped_mean_embeddings)):
                for j in range(len(mapped_mean_embeddings[0])):
                    scatter = ax.scatter(mapped_mean_embeddings[i][j,0], mapped_mean_embeddings[i][j,1], color=self.colors[j], marker=markers[i], label=labels[i])
                    if j==0:
                        handles.append(scatter)
        elif plot_dim==3:
            ax = fig.add_subplot(2,2,1, projection='3d')
            for i in range(len(mapped_mean_embeddings)):
                for j in range(len(mapped_mean_embeddings[0])):
                    scatter = ax.scatter(mapped_mean_embeddings[i][j,0], mapped_mean_embeddings[i][j,1], mapped_mean_embeddings[i][j,2], color=self.colors[j], marker=markers[i], label=labels[i])
                    if j==0:
                        handles.append(scatter)


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if plot_dim==3:
            ax.set_zlabel('Z')
        ax.set_title(f'{plot_dim}D Embeddings of NonAlcohol & Alcohol')
        ax.grid(True)


        # 凡例を追加
        ax.legend(handles, labels)

        # 各感情ごとにデータを収集
        emotion_embeddings = []
        for i in range(12):
            emotion_embeddings.append(np.vstack([mapped_mean_embeddings[j][i] for j in range(len(mapped_mean_embeddings))]))

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
        handles= dict()


        #プロット
        if plot_dim ==2:
            ax = fig.add_subplot(2,2,2)
            handle_cluster_flags = np.zeros(n_clusters)
            # 各感情のデータポイントをクラスタごとにプロット
            for i, embedding in enumerate(emotion_embeddings):
                for j in range(len(mapped_mean_embeddings)):
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
                for j in range(len(mapped_mean_embeddings)):
                    scatter = ax.scatter(embedding[j, 0], embedding[j, 1], embedding[j, 2],  color=emotion_colors[i], label=f'Cluster {emotion_labels[i]}' if j == 0 else "")
                    if handle_cluster_flags[emotion_labels[i]] == 0:
                        handles[f'Cluster {emotion_labels[i]}'] = scatter
                        handle_cluster_flags[emotion_labels[i]] = 1
                    ax.scatter(embedding[j, 0], embedding[j, 1],embedding[j, 2],  edgecolor=emotion_colors[i], facecolor='none')

        # グラフの装飾
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if plot_dim==3:
            ax.set_zlabel('Y')
        ax.set_title(f'{plot_dim}D Embeddings with {n_clusters} Clusters')
        ax.grid(True)

        #感情が属しているクラス分類
        cluster_emotions = {cluster: [] for cluster in range(n_clusters)}
        for i, emotion in enumerate(self.emotions):
            cluster_emotions[emotion_labels[i]].append(emotion)
        print("\nCluster Emotions:")
        for cluster, emotions_ in cluster_emotions.items():
            print(f"Cluster {cluster}: {emotions_}")

        np.array(mapped_mean_embeddings).shape
        #感情間の距離の表示
        embeddings_diff = mapped_mean_embeddings[0] - mapped_mean_embeddings[1]
        distances = [(diff[0]**2 + diff[1]**2)**0.5 for diff in embeddings_diff]
        # 棒グラフの作成
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(2,2,2)
        ax.bar(self.emotions, distances, color='skyblue')
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Distance')
        ax.set_title(f'NonAlcohol->NonAlcohol Mean: Distance for Each Emotion')
        ax.set_xticks(range(len(self.emotions)))  # ティックの位置を設定
        ax.set_xticklabels(self.emotions, rotation=45, ha='right')
        ax.grid(axis='y')

        plt.show()
        mean_distances_nonal_nonal = distances
        return mean_distances_nonal_nonal