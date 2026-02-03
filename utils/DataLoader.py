import os
import pandas as pd


class DataLoader:

    @staticmethod
    def load_data(path):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def save_prepared_df(df_prepared, directory="cache", path="solves_marked.pkl"):
        os.makedirs(directory, exist_ok=True)
        path_joined = os.path.join(directory, path)
        df_prepared.to_pickle(path_joined)

    @staticmethod
    def load_prepared_df(directory="cache", path="solves_marked.pkl"):
        path_joined = os.path.join(directory, path)
        return pd.read_pickle(path_joined)