

class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_data(path):
        df = pd.read_csv(path)
        return df