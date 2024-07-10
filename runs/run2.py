import numpy as np
from src.Preprocess import Preprocess
from src.CalculateDissimilarity import CalculateDissimilarity
from src.GWOptimalTransfer import GWOptimalTransfer
from src.extract_paths import extract_paths
from src.Cluster import Cluster

def run2(
        folder_path:str,
        qualia_color:dict,
        iter_num:int,
        n_clusters:int):
    #init
    emotions = list(qualia_color.keys())

    paths = extract_paths(folder_path=folder_path)
    print(f'paths{len(paths)}')
    subjects, preprocessed_dfs = run2_preprocess(paths=paths)
    print(f'subjects:{len(subjects)}')
    print(f'preprocessed_dfs:{np.array(preprocessed_dfs).shape}')
    dissimilarity_matrices, original_embeddings = run2_calculate_dissimilarity(
        preprocessed_dfs=preprocessed_dfs,
        emotions=emotions,
        plot_dim=2
    )
    print(f'dissimilarity_matrices:{dissimilarity_matrices.shape}')
    print(f'original_embeddings: {original_embeddings.shape}')

    optimal_P, optimal_Q, all_embeddings = run2_gwot(
        subjects=subjects,
        emotions=emotions,
        dissimilarity_matrices=dissimilarity_matrices,
        original_embeddings=original_embeddings,
        distance_plot_flag=False,
        iter_num=iter_num
    )
    cluster_emotions = run2_cluster(
        all_embeddings=all_embeddings,
        qualia_color=qualia_color,
        n_clusters=n_clusters
    )

    return optimal_P, optimal_Q, original_embeddings, all_embeddings, cluster_emotions


def run2_preprocess(
    paths:list
):
    subjects = list()
    preprocessed_dfs = list()
    for path in paths:
        #pathで指定された被験者の実験データを前処理して出力
        preprocess = Preprocess(path=path)
        #subject nameの抽出
        subject_name = preprocess.return_subject_name()
        subjects.append(subject_name)
        sim_list, comparison_video_pairs = preprocess.extract_cols()
        preprocessed_df = preprocess.output_data(sim_list, comparison_video_pairs)
        #保存
        preprocessed_dfs.append(preprocessed_df)
    
    #subjects: list len=9
    #preprocessed_dfs: 9(subjects) * 66(rows) * 2(columns)
    return subjects, preprocessed_dfs

def run2_calculate_dissimilarity(
        preprocessed_dfs,
        emotions,
        plot_dim:int=2
):
    cal_dissim = CalculateDissimilarity(emotions=emotions)
    dissimilarity_matrices = list()
    original_embeddings = list()
    for df in preprocessed_dfs:
        dissimilarity_matrix = cal_dissim.dissimilarity_matrix(
            preprocess_df=df
        )
        original_embedding = cal_dissim.original_embedding(
            dissimilarity_matrix=dissimilarity_matrix,
            plot_dim=plot_dim
        )
        #save
        dissimilarity_matrices.append(dissimilarity_matrix)
        original_embeddings.append(original_embedding)

    #dissimilarity_matrices: 9(subjects) * 12 * 12
    #original_embeddings: 9(subjects) * 12 * 2(plot_dim)
    return np.array(dissimilarity_matrices), np.array(original_embeddings)

def run2_gwot(
    subjects,
    emotions,
    dissimilarity_matrices,
    original_embeddings,
    distance_plot_flag,
    iter_num,
    colors=None,
):
    gwot = GWOptimalTransfer(emotions=emotions)
    all_embeddings = dict()
    all_distances = dict()

    #最初の埋め込みだけは最適輸送せずに直接埋め込む
    X_dissimilarity_matrix = dissimilarity_matrices[0]
    X_embedding = original_embeddings[0]
    all_embeddings[subjects[0]] = X_embedding

    #それ以降は最適輸送処理を施す
    for i in range(1, dissimilarity_matrices.shape[0]):
        Y_dissimilarity_matrix = dissimilarity_matrices[i]
        Y_embedding = original_embeddings[i]
        optimal_P, optimal_Q, personal_embeddings = gwot.run_personal_gwot(
            X_dissimilarity_matrix=X_dissimilarity_matrix,
            Y_dissimilarity_matrix=Y_dissimilarity_matrix,
            X_embedding=X_embedding,
            Y_embedding=Y_embedding,
            iter_num=iter_num
        )
        #save
        all_embeddings[subjects[i]] = personal_embeddings[1]
    
    return optimal_P, optimal_Q, all_embeddings


def run2_cluster(
        n_clusters,
        all_embeddings,
        qualia_color):
    cluster = Cluster(
        all_embeddings=all_embeddings,
        qualia_color=qualia_color
    )
    cluster_emotions = cluster.plot_all(n_clusters=n_clusters)
    return cluster_emotions