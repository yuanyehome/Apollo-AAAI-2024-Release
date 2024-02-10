from datasets import load_from_disk


def load_train_datasets():
    return load_from_disk("train_datasets")


if __name__ == "__main__":
    dataset = load_train_datasets()
    import pdb
    pdb.set_trace()
