import numpy as np
from src.Preprocess import Preprocess
from src.CalculateDissimilarity import CalculateDissimilarity
from src.GWOptimalTransfer import GWOptimalTransfer
from src.extract_paths import extract_paths_from_2paths
import pandas as pd

def run1(
    non_al_folder_path:str,
    al_folder_path:str,
    iter_num:int,
    qualia_color:dict
):
    #init
    emotions = list(qualia_color.keys())
    colors = list(qualia_color.values())

    first_paths, second_paths = extract_paths_from_2paths(
        first_folder_path=non_al_folder_path,
        second_folder_path=al_folder_path
    )
    # print(f'first_paths:{len(first_paths)}, second_paths: {len(second_paths)}')
    subjects, all_preprocessed_dfs = run1_preprocess(
        first_paths=first_paths,
        second_paths=second_paths
    )
    # print(f'subjects:{len(subjects)}')
    # print(f'all_preprocessed_dfs:{np.array(all_preprocessed_dfs).shape}')
    all_dissimilarity_matrices, all_original_embeddings = run1_calculate_dissimilarity(
        all_preprocessed_dfs=all_preprocessed_dfs,
        emotions=emotions,
        plot_dim=2,
    )
    print(f'all_dissimilarity_matrices:{all_dissimilarity_matrices.shape}')
    print(f'all_original_embeddings: {all_original_embeddings.shape}')

    optimal_P,  optimal_Q, all_embeddings, all_distances = run1_gwot(
        subjects=subjects,
        emotions=emotions,
        colors=colors,
        all_dissimilarity_matrices=all_dissimilarity_matrices,
        all_original_embeddings=all_original_embeddings,
        iter_num=iter_num,
        distance_plot_flag=True
    )
    distances_df = pd.DataFrame(all_distances, index=emotions)

    return optimal_P,  optimal_Q, all_original_embeddings, all_embeddings, distances_df


def run1_preprocess(
    first_paths:list,
    second_paths:list,
):
    """
    被験者をまとめて一気に前処理したいとき
    """
    all_preprocessed_dfs =list()
    subjects = list()
    for paths in [first_paths, second_paths]:
        preprocessed_dfs = list()
        for path in paths:
            #pathで指定された被験者の実験データを前処理して出力
            preprocess = Preprocess(path=path)
            if paths == first_paths:
                subject_name = preprocess.return_subject_name()
                subjects.append(subject_name)
            sim_list, comparison_video_pairs = preprocess.extract_cols()
            preprocessed_df = preprocess.output_data(sim_list, comparison_video_pairs)
            #保存
            preprocessed_dfs.append(preprocessed_df)
        all_preprocessed_dfs.append(preprocessed_dfs)
    
    #size = 9subjects * 66rows * 2columns
    return subjects, all_preprocessed_dfs


def run1_calculate_dissimilarity(
    all_preprocessed_dfs,
    emotions,
    plot_dim:int=2,
):
    """
    非類似度行列を計算し埋め込みベクトルを獲得
    """
    cal_dissim = CalculateDissimilarity(emotions=emotions)
    all_dissimilarity_matrices = list()
    all_original_embeddings = list()
    for preprocessed_dfs in all_preprocessed_dfs:
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
        #save
        all_dissimilarity_matrices.append(dissimilarity_matrices)
        all_original_embeddings.append(original_embeddings)

    #all_dissimilarity_matrices: 2(non_al/al) * 9(subjects) * 12 * 12
    #all_original_embeddings: 2(non_al/al) * 9(subjects) * 12 * 2(plot_dim)
    return np.array(all_dissimilarity_matrices), np.array(all_original_embeddings)

def run1_gwot(
        subjects,
        emotions,
        all_dissimilarity_matrices,
        all_original_embeddings,
        distance_plot_flag,
        iter_num,
        colors=None,
        ):
    gwot = GWOptimalTransfer(emotions=emotions)
    all_embeddings = dict()
    all_distances = dict()

    for i in range(len(subjects)):
        X_dissimilarity_matrix = all_dissimilarity_matrices[0][i]
        Y_dissimilarity_matrix = all_dissimilarity_matrices[1][i]
        X_embedding = all_original_embeddings[0][i]
        Y_embedding = all_original_embeddings[1][i]

        optimal_P, optimal_Q, personal_embeddings = gwot.run_personal_gwot(
            X_dissimilarity_matrix=X_dissimilarity_matrix,
            Y_dissimilarity_matrix=Y_dissimilarity_matrix,
            X_embedding=X_embedding,
            Y_embedding=Y_embedding,
            iter_num=iter_num
        )
        #save
        all_embeddings[subjects[i]] = personal_embeddings

        #distanceをプロットする場合
        if distance_plot_flag:
            distances = gwot.plot_personal_embedding_and_distance(
                personal_embeddings=personal_embeddings,
                subject=subjects[i],
                colors=colors
            )
            all_distances[subjects[i]] = distances

    return optimal_P, optimal_Q, all_embeddings, all_distances

