import pandas as pd
from datasets import load_dataset


def load_and_process_osm_data(dataset_name="ns2agi/antwerp-osm-navigator"):
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    print("\nDataset structure:")
    print(dataset)

    train_split = dataset["train"]
    print("\nShowing first 5 rows of the 'train' split:")
    print(train_split[:5])

    print("\nFiltering for 'node' types...")
    node_dataset = train_split.filter(lambda example: example["type"] == "node")
    print(f"Found {len(node_dataset)} nodes.")
    print("Showing first 5 nodes:")
    print(node_dataset[:5])

    print("\nConverting the first 100 nodes to a pandas DataFrame...")
    df = node_dataset.select(range(100)).to_pandas()

    print("\nPandas DataFrame head:")
    print(df.head())

    return df


if __name__ == "__main__":
    osm_dataframe = load_and_process_osm_data()
    print("\nScript finished.")
