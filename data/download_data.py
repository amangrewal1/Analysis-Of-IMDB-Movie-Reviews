from datasets import load_dataset

def download_imdb_dataset():
    dataset = load_dataset('imdb')
    return dataset

if __name__ == "__main__":
    download_imdb_dataset()
