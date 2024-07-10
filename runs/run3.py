from src.EmotionAnalysis import EmotionAnalysis


def run3(
    qualia_color,
    non_al_folder_path,
    al_folder_path,
    plot_dim,
    title,
    labels,
    iter_num=1000,
):
    ea = EmotionAnalysis(qualia_color=qualia_color)
    all_distances = ea.process1(
    non_al_folder_path=non_al_folder_path,
    al_folder_path=al_folder_path,
    plot_dim=plot_dim,
    iter_num=iter_num,
    title=title,
    labels=labels
    )

    return all_distances
