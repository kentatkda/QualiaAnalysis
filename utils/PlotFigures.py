import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PlotFigures:

    def __init__(self, fig, plot_dim, qualia_color):
        self.qualia_color = qualia_color
        self.emotions = list(self.qualia_color.keys())
        self.colors = list(self.qualia_color.values())
        self.fig = fig
        self.plot_dim = plot_dim

    def plot_original_embeddings(self, all_original_embeddings_list:list, attributes:list, plot_dim:int ):
        self.sample_size = len(all_original_embeddings_list[0])
        if plot_dim == 3:
            # fig = plt.figure(figsize=(10, 8))
            for i, original_embeddings_list in enumerate(all_original_embeddings_list):
                ax = self.fig.add_subplot(2, 2, i+1, projection='3d')
                for embedding in original_embeddings_list:
                    embedding_df = pd.DataFrame(embedding, index=self.emotions, columns=['Dim1', 'Dim2', 'Dim3'])
                    colors = [self.qualia_color[emotion] for emotion in embedding_df.index]

                    ax.scatter(embedding_df['Dim1'], embedding_df['Dim2'], embedding_df['Dim3'], c=colors)

                ax.set_xlabel('Dim1')
                ax.set_ylabel('Dim2')
                ax.set_zlabel('Dim3')
                ax.set_title(f'{attributes[i]} Embedding of Emotional Categories in 3D Space')

            # plt.suptitle(f'Non_Al / Al Embedding of Emotional Categories')
            # plt.tight_layout()
            # plt.show()

        elif plot_dim == 2:
            # fig = plt.figure(figsize=(10, 8))
            for i, original_embeddings_list in enumerate(all_original_embeddings_list):
                ax = self.fig.add_subplot(2, 2, i+1)
                for embedding in original_embeddings_list:
                    embedding_df = pd.DataFrame(embedding, index=self.emotions, columns=['Dim1', 'Dim2'])
                    colors = [self.qualia_color[emotion] for emotion in embedding_df.index]

                    ax.scatter(embedding_df['Dim1'], embedding_df['Dim2'], c=colors)

                ax.set_xlabel('Dim1')
                ax.set_ylabel('Dim2')

                ax.set_title(f'{attributes[i]} Embedding of Emotional Categories')

        return self.fig

    def plot_optimal_transfer_embeddings(self, X_embeddings_combined, Y_optimal_mapped):
        if self.plot_dim == 3:
            # 3次元プロットの準備
            # fig = plt.figure(figsize=(8, 6))
            ax = self.fig.add_subplot(2, 2, 4, projection='3d')

            # 座標をプロット
            colors_combined = self.colors * self.sample_size
            ax.scatter(Y_optimal_mapped[0], Y_optimal_mapped[1], Y_optimal_mapped[2], c=colors_combined, marker='x', label='optimal_mapped_Y')

            #元埋め込みも同様にプロット
            colors_combined = self.colors * self.sample_size
            embedding_df = pd.DataFrame(X_embeddings_combined.T, index=colors_combined, columns=['Dim1', 'Dim2', 'Dim3'])
            ax.scatter(embedding_df['Dim1'], embedding_df['Dim2'], embedding_df['Dim3'], c=colors_combined, marker='o', label='raw_X')


            # 軸ラベルの設定
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # グラフ表示
            # plt.legend()
            # plt.show()

        elif self.plot_dim == 2:

            # 2次元プロットの準備
            # fig = plt.figure(figsize=(12, 9))
            ax = self.fig.add_subplot(2, 2, 4)

            # 座標をプロット
            colors_combined = self.colors * self.sample_size
            ax.scatter(Y_optimal_mapped[0], Y_optimal_mapped[1], c=colors_combined, marker='x',s=15, label='optimal_mapped_Y')

            #元埋め込みも同様にプロット
            colors_combined = self.colors * self.sample_size
            embedding_df = pd.DataFrame(X_embeddings_combined.T, index=colors_combined, columns=['Dim1', 'Dim2'])
            ax.scatter(embedding_df['Dim1'], embedding_df['Dim2'], c=colors_combined, marker='o',s=15, label='raw_X')


            # 軸ラベルの設定
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # グラフ表示
            # plt.legend()
            # plt.show()
        return self.fig

    def plot_optimal_P_heatmap(self, optimal_P):

        #optimal_Pの可視化part2
        ax = self.fig.add_subplot(2, 2, 3)
        sns.heatmap(optimal_P, cmap='Reds')
        ax.set_title('Y→X heatmap of optimal P matrix')
        ax.set_xlabel('Y Indices')
        ax.set_ylabel('X Indices')

        return self.fig