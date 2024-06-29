import glob
import os

def extract_paths(al_folder_path:str, non_al_folder_path:str, comparing_pairs:tuple):

    non_al_files = glob.glob(os.path.join(non_al_folder_path, '*.csv'))
    al_files = glob.glob(os.path.join(al_folder_path, '*.csv'))

    if comparing_pairs==('al', 'non_al') or comparing_pairs==('non_al', 'al'):
        first_qualia_paths, second_qualia_paths = (non_al_files, al_files)

    elif comparing_pairs==('al', 'al'):
        half_len = len(al_files) // 2
        first_qualia_paths, second_qualia_paths = (al_files[:half_len], al_files[half_len: 2*half_len])

    elif comparing_pairs==('non_al', 'non_al'):
        half_len = len(non_al_files) // 2
        first_qualia_paths, second_qualia_paths = (non_al_files[:half_len], non_al_files[half_len: 2*half_len])

    else:
        print(f'this comparing_pairs: {comparing_pairs} is not valid.')

    return first_qualia_paths, second_qualia_paths
