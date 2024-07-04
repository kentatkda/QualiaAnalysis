import numpy as np
import os
import glob
from utils.Preprocess import Preprocess
from utils.CalculateDissimilarity import CalculateDissimilarity
from utils.GenerateSyntheticData import GenerateSyntheticData
from utils.GWOptimalTransfer import GWOptimalTransfer
from utils.PlotFigures import PlotFigures
from utils.extract_paths import extract_paths
import matplotlib.pyplot as plt
import pandas as pd

def process_of_extract(
        non_al_folder_path,
        al_folder_path,
        comparing_pairs
):
    first_qualia_paths, second_qualia_paths = extract_paths(
        non_al_folder_path=non_al_folder_path,
        al_folder_path=al_folder_path,
        comparing_pairs=comparing_pairs
    )

    return first_qualia_paths, second_qualia_paths


def process_of_preprocess(
        first_qualia_paths:list,
        second_qualia_paths:list,
        sample_size:int=14,
        generate_syn:bool=False
):
    all_preprocessed_dfs =list()
    subjects = list()
    for paths in [first_qualia_paths, second_qualia_paths]:

        #前処理
        if generate_syn:
            """
            sample数が少なく疑似データを生成したい場合
            non_al_paths: 要素数１のlist想定、開発段階ではsample_andrew.csvのみ
            al_paths: 要素数１のlist想定、開発段階ではsample_takada.csvのみ
            """
            #pathの抽出
            path = paths[0]

            #前処理
            preprocess = Preprocess(path=path)
            sim_list, comparison_video_pairs = preprocess.extract_cols()
            preprocessed_df = preprocess.output_data(sim_list, comparison_video_pairs)
            #14個の疑似データ生成(計15samples)
            generate_synthetic_data = GenerateSyntheticData()
            preprocessed_dfs = generate_synthetic_data.generate_synthetic_data(preprocessed_df=preprocessed_df, increasing_sample_size=sample_size)

        else:
            """
            本番用で、疑似データ生成しない場合
            non_al_paths: 要素数15のlist想定
            al_paths: 要素数15のlist想定
            """
            preprocessed_dfs = list()
            for path in paths:
                #pathで指定された被験者の実験データを前処理して出力
                preprocess = Preprocess(path=path)
                if paths == first_qualia_paths:
                    subject_name = preprocess.return_subject_name()
                    subjects.append(subject_name)
                sim_list, comparison_video_pairs = preprocess.extract_cols()
                preprocessed_df = preprocess.output_data(sim_list, comparison_video_pairs)
                #保存
                preprocessed_dfs.append(preprocessed_df)
            all_preprocessed_dfs.append(preprocessed_dfs)
    
    return subjects, all_preprocessed_dfs

def process_of_dissimilarity(
    all_preprocessed_dfs:pd.DataFrame,
    qualia_color:dict,
    plot_dim:int,
    plot_in_space_flag:bool=False
):
    all_original_embeddings = list()
    all_dissimilarity_matrices = list()
    attributes = ['non_alcohol', 'alcohol']

    #類似度行列を計算し、それを基にノンアル、アルコールの場合の埋め込みベクトルをそれぞれ表示
    cal_dissim = CalculateDissimilarity(
        qualia_color=qualia_color)
    all_dissimilarity_matrices = cal_dissim.dissimilarity_matrices(
        all_preprocessed_dfs=all_preprocessed_dfs
        )
    all_original_embeddings = cal_dissim.original_embeddings(
        all_dissimilarity_matrices=all_dissimilarity_matrices, 
        plot_dim=plot_dim)
    
    if plot_in_space_flag:
        cal_dissim.plot_in_space(
            all_original_embeddings=all_original_embeddings,
            attributes=attributes,
            plot_dim=plot_dim
        )

    return all_dissimilarity_matrices, all_original_embeddings

def process_of_mean_dissimilarity(
    all_preprocessed_dfs:pd.DataFrame,
    qualia_color:dict,
    plot_dim:int,
    plot_in_space_flag:bool=False
):
    #類似度行列を計算し、それを基にノンアル、アルコールの場合の埋め込みベクトルをそれぞれ表示
    cal_dissim = CalculateDissimilarity(
        qualia_color=qualia_color)
    all_dissimilarity_matrices = cal_dissim.dissimilarity_matrices(
        all_preprocessed_dfs=all_preprocessed_dfs
        )
    
    #平均化
    mean_matrices = list()
    for dissimilarity_matrices in all_dissimilarity_matrices:
        mean_matrix = np.mean(dissimilarity_matrices, axis=0)
        mean_matrices.append([mean_matrix])

    #平均化された埋め込みの獲得
    original_mean_embeddings = cal_dissim.original_embeddings(
        all_dissimilarity_matrices=mean_matrices, 
        plot_dim=plot_dim)
    
    return mean_matrices, original_mean_embeddings



def process_of_gwot(
        all_original_embeddings:np.array,
        qualia_color:dict,
        max_iter:int,
        plot_optimal_flag:bool
):
    #gw最適輸送の実行
    X_embeddings = all_original_embeddings[0]
    Y_embeddings = all_original_embeddings[1]
    gwot = GWOptimalTransfer(
        X_embeddings=X_embeddings,
        Y_embeddings=Y_embeddings,
        qualia_color=qualia_color)
    X_embeddings_combined, Y_embeddings_combined = gwot.reshape_embeddings()
    optimal_P = gwot.gromov_wasserstein(
        X_embeddings_combined=X_embeddings_combined,
        Y_embeddings_combined=Y_embeddings_combined,
        max_iter=max_iter)
    optimal_Q = gwot.calcurate_Q(
        X_embeddings_combined=X_embeddings_combined,
        Y_embeddings_combined=Y_embeddings_combined,
        optimal_P=optimal_P)
    Y_optimal_mapped = gwot.calcurate_mapped_Y(
        optimal_Q=optimal_Q,
        optimal_P=optimal_P,
        Y_embeddings_combined=Y_embeddings_combined)
    #元々のXの埋め込みと最適輸送による輸送後のYの埋め込みを同時にプロット
    if plot_optimal_flag:
        gwot.plot_optimal(X_embeddings_combined=X_embeddings_combined, Y_optimal_mapped=Y_optimal_mapped)

    return optimal_P, optimal_Q, Y_optimal_mapped

def run_tmp(
    non_al_folder_path:str,
    al_folder_path:str,
    generate_syn:bool,
    plot_dim:int,
    comparing_pairs:tuple,
    qualia_color: dict,
    max_iter:int,
    plot_optimal_flag:bool,
    sample_size:int=14) -> None:

    subjects, all_preprocessed_dfs = process_of_preprocess(
        non_al_folder_path=non_al_folder_path,
        al_folder_path=al_folder_path,
        comparing_pairs=comparing_pairs,
        generate_syn=generate_syn
    )

    all_dissimilarity_matrices, all_original_embeddings = process_of_dissimilarity(
        all_preprocessed_dfs=all_preprocessed_dfs,
        qualia_color=qualia_color,
        plot_dim=plot_dim
    )

    optimal_P, optimal_Q, Y_mapped_embeddings = process_of_gwot(
        all_original_embeddings=all_original_embeddings,
        qualia_color=qualia_color,
        max_iter=max_iter,
        plot_optimal_flag=plot_optimal_flag
    )

    return all_preprocessed_dfs



def run(
        non_al_folder_path:str,
        al_folder_path:str,
        generate_syn:bool,
        plot_dim:int,
        comparing_pairs:tuple,
        qualia_color: dict,
        max_iter:int,
        sample_size:int=14) -> None:

    first_qualia_paths, second_qualia_paths = extract_paths(
        non_al_folder_path=non_al_folder_path,
        al_folder_path=al_folder_path,
        comparing_pairs=comparing_pairs
    )
    all_preprocessed_dfs =list()
    all_original_embeddings = list()
    all_dissimilarity_matrices = list()
    attributes = ['non_alcohol', 'alcohol']
    for paths in [first_qualia_paths, second_qualia_paths]:

        #前処理
        if generate_syn:
            """
            sample数が少なく疑似データを生成したい場合
            non_al_paths: 要素数１のlist想定、開発段階ではsample_andrew.csvのみ
            al_paths: 要素数１のlist想定、開発段階ではsample_takada.csvのみ
            """
            #pathの抽出
            path = paths[0]

            #前処理
            preprocess = Preprocess(path=path)
            sim_list, comparison_video_pairs = preprocess.extract_cols()
            preprocessed_df = preprocess.output_data(sim_list, comparison_video_pairs)
            #14個の疑似データ生成(計15samples)
            generate_synthetic_data = GenerateSyntheticData()
            preprocessed_dfs = generate_synthetic_data.generate_synthetic_data(preprocessed_df=preprocessed_df, increasing_sample_size=sample_size)

        else:
            """
            本番用で、疑似データ生成しない場合
            non_al_paths: 要素数15のlist想定
            al_paths: 要素数15のlist想定
            """
            preprocessed_dfs = list()
            subjects = list()
            for path in paths:
                #pathで指定された被験者の実験データを前処理して出力
                preprocess = Preprocess(path=path)
                if paths == first_qualia_paths:
                    subject_name = preprocess.return_subject_name()
                    subjects.append(subject_name)
                sim_list, comparison_video_pairs = preprocess.extract_cols()
                preprocessed_df = preprocess.output_data(sim_list, comparison_video_pairs)
                #保存
                preprocessed_dfs.append(preprocessed_df)
            all_preprocessed_dfs.append(preprocessed_dfs)

    #類似度行列を計算し、それを基にノンアル、アルコールの場合の埋め込みベクトルをそれぞれ表示
    cal_dissim = CalculateDissimilarity(
        all_preprocessed_dfs=all_preprocessed_dfs,
        qualia_color=qualia_color)
    all_dissimilarity_matrices = cal_dissim.dissimilarity_matrices()
    all_original_embeddings = cal_dissim.original_embeddings(
        all_dissimilarity_matrices=all_dissimilarity_matrices, 
        plot_dim=plot_dim)

    #ノンアル/アルコール時の埋め込みベクトルを横並びに表示
    # cal_dissim.plot_in_space(all_original_embeddings_list=all_original_embeddings_list , plot_dim=plot_dim, attributes=attributes)

    #gw最適輸送の実行
    X_embeddings = all_original_embeddings[0]
    Y_embeddings = all_original_embeddings[1]
    gwot = GWOptimalTransfer(
        X_embeddings=X_embeddings,
        Y_embeddings=Y_embeddings,
        qualia_color=qualia_color)
    X_embeddings_combined, Y_embeddings_combined = gwot.reshape_embeddings()
    optimal_P = gwot.gromov_wasserstein(
        X_embeddings_combined=X_embeddings_combined,
        Y_embeddings_combined=Y_embeddings_combined,
        max_iter=max_iter)
    return subjects, optimal_P
    #マッピング行列Pを表示
    # # gwot.plot_optimal_P_heatmap(optimal_P=optimal_P, optimal_P_identical=optimal_P_identical)

    # optimal_Q = gwot.calcurate_Q(
    #     X_embeddings_combined=X_embeddings_combined,
    #     Y_embeddings_combined=Y_embeddings_combined,
    #     optimal_P=optimal_P)
    # Y_optimal_mapped = gwot.calcurate_mapped_Y(
    #     optimal_Q=optimal_Q,
    #     optimal_P=optimal_P,
    #     Y_embeddings_combined=Y_embeddings_combined)
    # #元々のXの埋め込みと最適輸送による輸送後のYの埋め込みを同時にプロット
    # # gwot.plot_optimal(X_embeddings_combined=X_embeddings_combined, Y_optimal_mapped=Y_optimal_mapped)

    # #全てのグラフをまとめて表示
    # fig = plt.figure(figsize=(10,10))
    # plot_figures = PlotFigures(fig=fig, plot_dim=plot_dim, qualia_color=qualia_color)
    # fig1 = plot_figures.plot_original_embeddings(
    #     all_original_embeddings_list=all_original_embeddings_list,
    #     attributes=attributes,
    #     plot_dim=plot_dim)
    # fig2 = plot_figures.plot_optimal_P_heatmap(optimal_P=optimal_P)
    # fig3 = plot_figures.plot_optimal_transfer_embeddings(
    #     X_embeddings_combined=X_embeddings_combined,
    #     Y_optimal_mapped=Y_optimal_mapped)
    # plt.legend()
    # plt.show()