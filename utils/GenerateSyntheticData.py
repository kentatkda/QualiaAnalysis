import numpy as np


class GenerateSyntheticData:
    """
    データ数を疑似的に増やすためのクラス
    """
    def __init__(self) -> None:
        pass

    def generate_synthetic_data(self, preprocessed_df, increasing_sample_size):
        syn_dfs = [preprocessed_df]
        for num in range(increasing_sample_size):
            syn_df = preprocessed_df.copy()
            # similarity列に-1, 0, 1のランダム増減を加える
            adjustments = np.random.choice([-0.5, 0, 0.5], size=70)
            syn_df['similarity'] += adjustments
            syn_dfs.append(syn_df)
        return syn_dfs