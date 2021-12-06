import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path, delimiter=',', engine='python')
    return df

# def clean_dataset(dataset):


def main():
    dataset = load_dataset('data/csgo_round_snapshots.csv')
    # clean_dataset(dataset)
    print(dataset.info)

main()


