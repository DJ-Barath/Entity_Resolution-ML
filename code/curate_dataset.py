import os
from pathlib import Path
import pandas as pd

# ============================================================
# Configuration
# ============================================================


INPUT = Path("")   # Folder containing the original datasets
OUTPUT = Path("")   # Folder where curated datasets will be saved

INPUT_SEPT = '\t'
OUTPUT_SEPT = '\t'

def assign_dir(*, input = "Dataset", output = "Sample_Dataset"):
    global INPUT, OUTPUT
    INPUT = Path(input)
    OUTPUT = Path(output)

assign_dir()

# ============================================================
# Create output directory
# ============================================================

# this function call is expensive, it takes large time so avoid if not needed - mynote
def count_rows(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # subtract header
    

def curate_dataset(input_file, output_file, n):
    """
    Read a TSV dataset, take n records from it,
    and save the result as a TSV file.
    """

    curated_df = pd.read_csv(
        input_file,
        sep=INPUT_SEPT,
        nrows=n,
        dtype=str,
        keep_default_na=False
    )

    # if number of entries in original dataset is needed
    # print(f"{input_file}: {count_rows(input_file)} records found")

    # Take at most n records

    curated_df.to_csv(
        output_file,
        sep=OUTPUT_SEPT,
        index=False
    )

    print(f"Saved {len(curated_df)} records -> {output_file}")


# ============================================================
# Training datasets
# ============================================================

def curate_datasets(pathList, N) :

    # Define your nested folder structure
    folder_path = Path(OUTPUT)
    if not folder_path.exists():
        print("output directory does not exist!")
        # Create the structure safely
        folder_path.mkdir(parents=True, exist_ok=True)
        print("Output folder Created")

    for path in pathList:
        curate_dataset(
            INPUT / path,
            OUTPUT / path,
            N
        )

def main(N) :
    assign_dir(input="Dataset/train", output = "Sample_Dataset/train")
    trainList = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]
    curate_datasets(trainList, N)

    assign_dir(input="Dataset/test", output = "Sample_Dataset/test")
    testList = ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]
    curate_datasets(testList, N)
    print("\nCuration completed successfully.")

if __name__ == "__main__" :
    # Number of records to take from each source
    N = 1000
    main(N)