import pandas as pd


class Helper:
    @staticmethod
    def write_to_csv(data, filename, header=True):
        df = pd.DataFrame(data)
        df.to_csv(filename, header=header, index=False)


