from research_nonal import run_nonal_all
from research_nonal_al import run_nonal_al_all

if __name__ == "__main__":
  mode='only_nonal'
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
    original_embeddings, all_embeddings = run_nonal_all(
      folder_path=folder_path,
      qualia_color=qualia_color,
      iter_num=2,
      n_clusters=3
    )
  elif mode=='nonal_and_al':
    non_al_folder_path = './qualia_rawdata/non_alcohol/'
    al_folder_path = './qualia_rawdata/alcohol/'
    all_original_embeddings, all_embeddings, distances_df = run_nonal_al_all(
      non_al_folder_path=non_al_folder_path,
      al_folder_path=al_folder_path,
      iter_num=4,
      qualia_color=qualia_color
    )
