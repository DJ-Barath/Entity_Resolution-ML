
# -----------------------------
# Dataset locations
# -----------------------------
splitType = "train"
SOURCE_DIR = f"Sample_DataSet/{splitType}"

SOURCE_S1 =  SOURCE_DIR + f"/{splitType}_source1.tsv"
SOURCE_S2 =  SOURCE_DIR + f"/{splitType}_source2.tsv"
SOURCE_S3 =  SOURCE_DIR + f"/{splitType}_source3.tsv"

# Output file containing candidate pairs
OUTPUT_DIR = f"PreProcessedSample/{splitType}" 

OUTPUT_S1 =  OUTPUT_DIR + f"/{splitType}_source1.tsv"
OUTPUT_S2 =  OUTPUT_DIR + f"/{splitType}_source2.tsv"
OUTPUT_S3 =  OUTPUT_DIR + f"/{splitType}_source3.tsv"

# ----------------------------
# text Column names
# ---------------------------

TEXT_COLUMNS = [
    "business_name",
    "business_address",
    "country",
]

# -----------------------------
# initialization code
# -----------------------------

def init_Output_dir(output = "./defaultNormalizer/"):
    from pathlib import Path

    # creating Path object
    folder_path = Path(output)
    folder_path.mkdir(parents = True, exist_ok = True)

if __name__ == "__main__" :
    init_Output_dir(OUTPUT_DIR)

