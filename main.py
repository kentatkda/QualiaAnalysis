from run import run
import os
import glob

if __name__ == "__main__":
  non_al_folder_path = './qualia_rawdata/single_subject/non_alcohol/'
  al_folder_path = './qualia_rawdata/single_subject/alcohol/'
  # qualia_color = {
  #   'sadness': 'blue',
  #   'romance': 'pink',
  #   'awe': 'green',
  #   'aesthetic_appreciation': 'yellow',
  #   'amusement': 'brown',
  #   'guilt': 'black',
  #   'surprise': 'gray',
  #   'fear': 'violet',
  #   'anger': 'red',
  #   'empatic_pain': 'orange',
  #   'envy': 'purple',
  #   'joy': 'cyan'
  # }

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

  # qualia_color = {
  #   'sadness': 'blue',
  #   'romance': 'blue',
  #   'awe': 'green',
  #   'aesthetic_appreciation': 'yellow',
  #   'amusement': 'green',
  #   'guilt': 'red',
  #   'surprise': 'red',
  #   'fear': 'yellow',
  #   'anger': 'red',
  #   'empatic_pain': 'red',
  #   'envy': 'purple',
  #   'joy': 'blue'
  # }

  run(
    non_al_folder_path=non_al_folder_path,
    al_folder_path=al_folder_path,
    plot_dim=2,
    generate_syn=False,
    qualia_color=qualia_color,
    max_iter=500,
    comparing_pairs=('non_al', 'al')
  )
