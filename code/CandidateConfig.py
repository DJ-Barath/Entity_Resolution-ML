# ============================================================
# Candidate Blocking Configuration
# ============================================================

# -----------------------------
# Dataset locations
# -----------------------------
splitType = "train"
BASE_PATH = f"PreProcessedSample/{splitType}"

S1_PATH =  BASE_PATH + f"/{splitType}_source1.tsv"
S2_PATH =  BASE_PATH + f"/{splitType}_source2.tsv"
S3_PATH =  BASE_PATH + f"/{splitType}_source3.tsv"

# Output file containing candidate pairs
OUTPUT_PATH = BASE_PATH + "/candidate_pairs.tsv"


# -----------------------------
# Column names
# -----------------------------

ENTITY_ID_COLUMN = "entity_id"
NAME_COLUMN = "business_name"
ADDRESS_COLUMN = "business_address"
COUNTRY_COLUMN = "country"


# -----------------------------
# Embedding model
# -----------------------------

# Multilingual sentence-transformer.
#
# Good for multilingual entity/name/address data.
#
# You can replace this later with another multilingual
# SentenceTransformer model if required.
MODEL_NAME = "./Models/multilingual-MiniLM-L12-v2"
HUGGING_FACE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# to use cached model
MODEL_NAME = HUGGING_FACE_MODEL_NAME


# -----------------------------
# FAISS configuration
# -----------------------------

# ---  HNSW Configurations ---
# Number of connections per node in the graph
M = 32

# Number of candidates returned for each source record.
#
# Number of candidates retrieved from EACH FAISS index.
TOP_K = 20
TOP_K_PER_FEATURE = 20

# Batch size for embedding generation.
#
# Reduce this if GPU/RAM is limited.
BATCH_SIZE = 64



# ============================================================
# Similarity Weights
# ============================================================

NAME_WEIGHT = 0.50
ADDRESS_WEIGHT = 0.35
COUNTRY_WEIGHT = 0.15

# -----------------------------
# Blocking options
# -----------------------------

# Whether to remove candidates whose cosine similarity
# is below this value.
#
# For initial experiments, keeping this as None is safer
# because blocking should prioritize recall.
COMBINED_SIMILARITY_THRESHOLD = None
SIMILARITY_THRESHOLD = 0.8

# Example:
# SIMILARITY_THRESHOLD = 0.50

# -----------------------------
# Sources to process
# -----------------------------

SOURCE_FILES = {
    "S2": S2_PATH,
    "S3": S3_PATH,
}

# -----------------------------
# Candidate_Pairs
# -----------------------------

S1_INDEX = 0
MATCH_INDEX = 1
SIMILARITY = 2
RANK = 3

S1_NAME = 4
S1_ADDRESS = 5
S1_COUNTRY = 6

MATCH_NAME = 7
MATCH_ADDRESS = 8
MATCH_COUNTRY = 9

candidatePairs = [
    "s1_index",
    "match_index",
    "similarity",
    "rank", # starts with 1; 1 indicates highest matching of "match" candidate
    
    "s1_name",
    "s1_address",
    "s1_country",

    "match_name",
    "match_address",
    "match_country"
]

candidatePairsSave = [
    "s1_index",
    "match_index",
    "similarity",
    "rank",

    "s1_name",
    "s1_address",
    "s1_country",

    "match_name",
    "match_address",
    "match_country"
]

# ------------------------------------------
# Model installer from Hugging face
# ------------------------------------------

def installModel(model_name, model_output_directory = "./my_custom_model") :
    from huggingface_hub import snapshot_download

    # 1. Define your model name and the folder where you want to save it
    # eg : model_name = "all-MiniLM-L6-v2"

    # 2. Download the model to your specified directory
    snapshot_download(repo_id = model_name, local_dir = model_output_directory)

    # 3. Load the model from that specific directory
    # model = SentenceTransformer(model_output_directory)

if __name__ == "__main__" :
    from pathlib import Path
    model_folder = Path(MODEL_NAME)
    model_folder.mkdir(parents = True, exist_ok = True)
    installModel(HUGGING_FACE_MODEL_NAME, MODEL_NAME)
