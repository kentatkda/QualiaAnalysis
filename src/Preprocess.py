import pandas as pd
import numpy as np
import ast

class Preprocess:
    """
    生データに対して前処理を行うクラス
    """
    def __init__(self, path):
        self.qualia_dict = {
        '0059': 'sadness',
        '0283': 'romance',
        '0927': 'awe',
        '1071': 'aesthetic_appreciation',
        '1158': 'amusement',
        '1169': 'guilt',
        '1292': 'surprise',
        '1375': 'fear',
        '1703': 'anger',
        '1768': 'empatic_pain',
        '1791': 'envy',
        '2138': 'joy'
    }
        self.rawdata = pd.read_csv(path, sep=',')
        self.subject_id = self.rawdata['Prolific_ID*'][0]

    def return_subject_name(self):
        return self.subject_id

    def extract_video_num(self, video_name):
        a = video_name.split('/')[-1]
        file_num = a.split('.')[0]
        return file_num
    
    def extract_cols(self):

        #類似度の抽出
        sample_sim_list = self.rawdata.similarity.tolist()[7:77]

        #比較ビデオの抽出
        seq_list = ast.literal_eval(self.rawdata[:1].sequence[0])
        seq_video_num = [self.extract_video_num(video) for video in seq_list]
        seq_qualia_list = [self.qualia_dict[video_num] for video_num in seq_video_num]
        comparsion_video_pairs = [ (seq_qualia_list[index], seq_qualia_list[index+1]) for index in range(len(seq_qualia_list)-1)]
        # print(f'video pairs: {len(comparsion_video_pairs)}')
        return sample_sim_list, comparsion_video_pairs

    def output_data(self,sample_sim_list, comparsion_video_pairs):
        #df出力
        output_df = pd.DataFrame({'compared_video': comparsion_video_pairs, 'similarity': sample_sim_list})
        #compared_videoのカラムのユニーク化
        output_df = output_df.groupby('compared_video', as_index=False).agg({'similarity': 'mean'})
        return output_df