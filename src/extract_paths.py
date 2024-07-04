import glob
import os

def extract_paths(folder_path:str, sep:bool=False):
    paths = glob.glob(os.path.join(folder_path, '*.csv'))
    if sep:
        half_len = len(paths)//2
        first_qualia_paths, second_qualia_paths = (paths[:half_len], paths[half_len: 2*half_len])
        return first_qualia_paths, second_qualia_paths
    else:
        return paths

def extract_paths_from_2paths(first_folder_path:str, second_folder_path):
    first_paths = glob.glob(os.path.join(first_folder_path, '*.csv'))
    second_paths = glob.glob(os.path.join(second_folder_path, '*.csv'))
    return first_paths, second_paths

