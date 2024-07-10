from runs.run1 import run1
from runs.run2 import run2
from runs.run3 import run3

def run(mode):
    qualia_color = {
    'guilt': 'black',
    'empatic_pain': 'orange',
    'fear': 'violet',
    'anger': 'red',
    'envy': 'purple',
    'sadness': 'blue',
    'surprise': 'gray',
    'joy': 'cyan',
    'amusement': 'brown',
    'romance': 'pink',
    'aesthetic_appreciation': 'yellow',
    'awe': 'green',
    }

    if mode=='only_nonal':
        folder_path='./qualia_rawdata/non_alcohol/'
        original_embeddings, all_embeddings = run2(
            folder_path=folder_path,
            qualia_color=qualia_color,
            iter_num=1000,
            n_clusters=3
        )


    elif mode=='nonal_and_al':
        non_al_folder_path = './qualia_rawdata/non_alcohol/'
        al_folder_path = './qualia_rawdata/alcohol/'
        all_original_embeddings, all_embeddings, distances_df = run1(
            non_al_folder_path=non_al_folder_path,
            al_folder_path=al_folder_path,
            iter_num=4,
            qualia_color=qualia_color
        )
        return all_embeddings

    elif mode=='quolia_analysis_nonal_and_al':
        non_al_folder_path = './qualia_rawdata/non_alcohol/'
        al_folder_path = './qualia_rawdata/alcohol/'
        distances = run3(
            non_al_folder_path=non_al_folder_path,
            al_folder_path=al_folder_path,
            plot_dim=3,
            iter_num=1000,
            title='Embedding of all subjects',
            labels=['non_alcohol', 'alcohol']
        )

        return distances