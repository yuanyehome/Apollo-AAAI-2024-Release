from datasets import load_dataset, concatenate_datasets, DatasetDict


def download_train_datasets():
    bookcorpus = load_dataset("bookcorpus", split="train", trust_remote_code=True)
    wiki = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

    assert bookcorpus.features.type == wiki.features.type
    raw_datasets = concatenate_datasets([bookcorpus, wiki])
    return DatasetDict({"train": raw_datasets})


if __name__ == "__main__":
    train_datasets = download_train_datasets()
    print(train_datasets.num_rows)
    print(train_datasets.num_columns)
    print(train_datasets.column_names)
    print(train_datasets.keys())
    train_datasets.save_to_disk("train_datasets")
