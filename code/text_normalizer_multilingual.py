# ```python

import pandas as pd
import re
import unicodedata
from pathlib import Path

import normalizerConfig as config

# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "s1": {
        "input": config.SOURCE_S1,
        "output": config.OUTPUT_S1,
    },
    "s2": {
        "input": config.SOURCE_S2,
        "output": config.OUTPUT_S2,
    },
    "s3": {
        "input": config.SOURCE_S3,
        "output": config.OUTPUT_S3,
    },
}

# Set this to None to process the entire file at once.
# For large datasets, use a chunk size such as 100_000.
CHUNK_SIZE = None

# Columns to normalize.
# Change these according to your dataset.
TEXT_COLUMNS = config.TEXT_COLUMNS


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def unicode_normalize(text):
    """
    1. Unicode normalization.

    NFKC converts different Unicode representations into
    a canonical compatible representation.

    This is language-independent.
    """

    if not isinstance(text, str):
        return ""

    return unicodedata.normalize("NFKC", text)


def whitespace_normalize(text):
    """
    2. Whitespace normalization.

    Examples:

        '  Microsoft   Corp  '
            ->
        'Microsoft Corp'

    Also handles tabs and newlines.
    """

    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


def punctuation_normalize(text):
    """
    3. Punctuation normalization.

    Replaces punctuation with spaces instead of simply
    deleting it, preventing words from being joined.

    Example:

        'ABC, Inc.'
            ->
        'ABC Inc'

    Unicode letters and numbers are preserved.
    """

    if not isinstance(text, str):
        return ""

    # \w keeps Unicode letters/numbers in Python 3.
    text = re.sub(r"[^\w\s]", " ", text)

    # Clean up spaces introduced by punctuation removal.
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text):
    """
    Complete multilingual normalization pipeline.

    Order is important:

        Unicode
            ↓
        Whitespace
            ↓
        Punctuation
            ↓
        Whitespace again
    """

    text = unicode_normalize(text)
    text = whitespace_normalize(text)
    text = punctuation_normalize(text)

    return text


# ============================================================
# DATASET NORMALIZATION
# ============================================================

def normalize_dataframe(df, text_columns):
    """
    Normalize selected columns while preserving the
    original dataset columns.

    For example:

        name
        address
        country

    become:

        name
        address
        country

        name_normalized
        address_normalized
        country_normalized
    """

    for column in text_columns:

        if column not in df.columns:
            print(
                f"Warning: column '{column}' "
                f"not found. Skipping."
            )
            continue

        normalized_column = f"{column}_normalized"

        df[normalized_column] = (
            df[column]
            .fillna("")
            .astype(str)
            .map(normalize_text)
        )

    return df


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def normalize_dataset(
    input_file,
    output_file,
    text_columns,
    chunk_size=None,
):
    """
    Normalize one TSV dataset.

    If chunk_size is None:
        Entire dataset is loaded into memory.

    If chunk_size is specified:
        Dataset is processed incrementally.
        This is better for large datasets.
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_file}"
        )

    # --------------------------------------------------------
    # Small/medium dataset
    # --------------------------------------------------------

    if chunk_size is None:

        df = pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )

        print(
            f"{input_file}: "
            f"loaded {len(df):,} rows"
        )

        df = normalize_dataframe(
            df,
            text_columns,
        )

        df.to_csv(
            output_path,
            sep="\t",
            index=False,
            encoding="utf-8",
        )

        print(
            f"{output_file}: "
            f"saved {len(df):,} rows"
        )

        return

    # --------------------------------------------------------
    # Large dataset - process in chunks
    # --------------------------------------------------------

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):

        chunk = normalize_dataframe(
            chunk,
            text_columns,
        )

        chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            encoding="utf-8",
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        first_chunk = False

        total_rows += len(chunk)

        print(
            f"{input_file}: "
            f"processed {total_rows:,} rows"
        )

    print(
        f"{output_file}: "
        f"saved {total_rows:,} rows"
    )


# ============================================================
# PROCESS S1, S2 AND S3
# ============================================================

def main():

    for dataset_name, config in DATASETS.items():

        print("\n" + "=" * 60)
        print(f"Processing {dataset_name.upper()}")
        print("=" * 60)

        normalize_dataset(
            input_file=config["input"],
            output_file=config["output"],
            text_columns=TEXT_COLUMNS,
            chunk_size=CHUNK_SIZE,
        )


if __name__ == "__main__":
    main()

"""
### Directory structure

Put the files like this:

```text
project/
│
├── normalize.py
│
├── s1.tsv
├── s2.tsv
└── s3.tsv
```

Run:

```bash
python normalize.py
```

You will get:

```text
project/
│
├── s1.tsv
├── s2.tsv
├── s3.tsv
│
├── s1_normalized.tsv
├── s2_normalized.tsv
└── s3_normalized.tsv
```

### Example

Suppose `s1.tsv` contains:

```text
entityId	name	address	country
E001	Microsoft   Corporation	One Microsoft Way, Redmond, WA	United States
E002	微软公司	北京市海淀区	China
E003	மைக்ரோசாப்ட்	சென்னை, தமிழ்நாடு	India
```

The output retains the original columns and adds:

```text
entityId	name	address	country	name_normalized	address_normalized	country_normalized
E001	Microsoft   Corporation	One Microsoft Way, Redmond, WA	United States	Microsoft Corporation	One Microsoft Way Redmond WA	United States
E002	微软公司	北京市海淀区	China	微软公司	北京市海淀区	China
E003	மைக்ரோசாப்ட்	சென்னை, தமிழ்நாடு	India	மைக்ரோசாப்ட்	சென்னை தமிழ்நாடு	India
```

Notice that **Chinese, Tamil, or other scripts are not transliterated or converted to English**.

That is intentional.

For your next stage, you can take:

```text
name_normalized
address_normalized
country_normalized
```

and generate **multilingual embeddings**, then use **FAISS/HNSW to retrieve candidate records**.

One recommendation: if your datasets are very large, set:

```python
CHUNK_SIZE = 100_000
```

so the program doesn't load the entire TSV into RAM. The normalization itself is stateless, so chunk processing is safe.
"""