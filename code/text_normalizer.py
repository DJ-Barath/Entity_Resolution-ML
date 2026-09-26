""" 
this works only for english text. Generally this level of text normalization is not required.

"""

# ```python
"""
Entity Resolution - TSV Normalization Pipeline

Input:
    source.tsv

Output:
    normalized_output.tsv

Configuration:
    normalization_config.json

The original columns are preserved.
Normalized/derived columns are added alongside them.
"""

import pandas as pd
import re
import json
import unicodedata
from pathlib import Path
import sys

# ============================================================
# FILE CONFIGURATION
# ============================================================

source = "source.tsv"
normalized_output = "normalized_output.tsv"

def assign_path(src, out) :
    global source, normalized_output
    source = src
    normalized_output = out
    print(f"changed Path-- src : {src} \n\t ouput : {normalized_output}")

# Stores the exact normalization rules used
normalization_config_file = "normalization_config.json"


# ============================================================
# NORMALIZATION DICTIONARIES
# ============================================================

# Company-name abbreviations
COMPANY_ABBREVIATIONS = {
    "corp": "corporation",
    "corp.": "corporation",
    "co": "company",
    "co.": "company",
    "inc": "incorporated",
    "inc.": "incorporated",
    "ltd": "limited",
    "ltd.": "limited",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "plc": "public limited company",
    "pvt": "private",
    "pvt.": "private",
    "intl": "international",
    "intl.": "international",
    "tech": "technology",
    "tech.": "technology",
}


# Address abbreviations
ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "st.": "street",
    "rd": "road",
    "rd.": "road",
    "ave": "avenue",
    "ave.": "avenue",
    "av": "avenue",
    "av.": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "hwy": "highway",
    "hwy.": "highway",
    "pkwy": "parkway",
    "pkwy.": "parkway",
    "ct": "court",
    "ct.": "court",
    "pl": "place",
    "pl.": "place",
    "sq": "square",
    "sq.": "square",
    "bldg": "building",
    "bldg.": "building",
    "fl": "floor",
    "fl.": "floor",
    "apt": "apartment",
    "apt.": "apartment",
    "no": "number",
    "no.": "number",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}


# Country normalization
COUNTRY_MAPPING = {
    # India
    "india": "IN",
    "ind": "IN",
    "in": "IN",
    "republic of india": "IN",

    # United States
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",

    # United Kingdom
    "united kingdom": "GB",
    "great britain": "GB",
    "uk": "GB",
    "gb": "GB",

    # Canada
    "canada": "CA",
    "can": "CA",
    "ca": "CA",

    # Australia
    "australia": "AU",
    "aus": "AU",
    "au": "AU",

    # Germany
    "germany": "DE",
    "deu": "DE",
    "de": "DE",

    # France
    "france": "FR",
    "fra": "FR",
    "fr": "FR",

    # Japan
    "japan": "JP",
    "jpn": "JP",
    "jp": "JP",

    # China
    "china": "CN",
    "chn": "CN",
    "cn": "CN",

    # Singapore
    "singapore": "SG",
    "sgp": "SG",
    "sg": "SG",

    # UAE
    "united arab emirates": "AE",
    "uae": "AE",
    "are": "AE",
    "ae": "AE",
}


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def unicode_normalize(value):
    """Apply Unicode NFKC normalization."""

    if pd.isna(value):
        return ""

    return unicodedata.normalize("NFKC", str(value))


def lowercase(value):
    """Convert text to lowercase."""

    if not value:
        return ""

    return value.lower()


def normalize_whitespace(value):
    """Convert repeated whitespace to a single space."""

    if not value:
        return ""

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_punctuation(value):
    """
    Replace punctuation with spaces.

    Example:
        ABC, Inc. -> ABC Inc
    """

    if not value:
        return ""

    return re.sub(r"[^\w\s]", " ", value)


def normalize_basic(value):
    """
    Complete basic text normalization.
    """

    value = unicode_normalize(value)
    value = lowercase(value)
    value = normalize_punctuation(value)
    value = normalize_whitespace(value)

    return value


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(value):
    """Convert normalized text into tokens."""

    if not value:
        return []

    return value.split()


# ============================================================
# COMPANY NAME NORMALIZATION
# ============================================================

def normalize_company_abbreviations(tokens):

    result = []

    for token in tokens:

        if token in COMPANY_ABBREVIATIONS:
            result.append(COMPANY_ABBREVIATIONS[token])
        else:
            result.append(token)

    return result


def extract_legal_suffix(tokens):

    """
    Extract common company legal suffixes.

    Example:
        Microsoft Corporation

        core_name:
            microsoft

        legal_suffix:
            corporation
    """

    legal_suffixes = {
        "corporation",
        "company",
        "incorporated",
        "limited",
        "private",
        "plc",
        "llc",
        "llp",
    }

    if not tokens:
        return "", ""

    # Check up to the last 4 tokens
    for length in range(
        min(4, len(tokens)),
        0,
        -1
    ):

        suffix = " ".join(tokens[-length:])

        if suffix in legal_suffixes:

            core = " ".join(tokens[:-length])

            return core.strip(), suffix.strip()

    return " ".join(tokens), ""


def normalize_company_name(value):

    # Basic normalization
    value = normalize_basic(value)

    # Tokenize
    tokens = tokenize(value)

    # Expand company abbreviations
    tokens = normalize_company_abbreviations(tokens)

    # Reconstruct normalized name
    normalized_name = " ".join(tokens)

    # Extract legal suffix
    core_name, legal_suffix = extract_legal_suffix(tokens)

    return {
        "name_normalized": normalized_name,
        "name_core": core_name,
        "name_legal_suffix": legal_suffix,
        "name_tokens": " ".join(tokens)
    }


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address_abbreviations(tokens):

    result = []

    for token in tokens:

        if token in ADDRESS_ABBREVIATIONS:
            result.append(
                ADDRESS_ABBREVIATIONS[token]
            )
        else:
            result.append(token)

    return result


def normalize_address_numbers(value):

    """
    Normalize address number formatting.

    Example:
        #123 Main St
        -> 123 Main St

    Important:
        We do NOT modify values such as 12A because
        they may represent meaningful building/unit numbers.
    """

    if not value:
        return ""

    # Remove # only when followed by a number
    value = re.sub(
        r"#\s*(?=\d)",
        "",
        value
    )

    return value


def normalize_address(value):

    # Unicode
    value = unicode_normalize(value)

    # Lowercase
    value = lowercase(value)

    # Address number normalization
    value = normalize_address_numbers(value)

    # Punctuation
    value = normalize_punctuation(value)

    # Whitespace
    value = normalize_whitespace(value)

    # Tokenize
    tokens = tokenize(value)

    # Expand address abbreviations
    tokens = normalize_address_abbreviations(tokens)

    normalized_address = " ".join(tokens)

    return {
        "address_normalized": normalized_address,
        "address_tokens": " ".join(tokens)
    }


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def normalize_country(value):

    if pd.isna(value):
        return ""

    value = unicode_normalize(value)
    value = lowercase(value)
    value = normalize_whitespace(value)

    return COUNTRY_MAPPING.get(
        value,
        value.upper()
    )


# ============================================================
# LOAD TSV
# ============================================================

def load_dataset(file_path):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    # Explicitly read as TSV
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return df


# ============================================================
# SAVE TSV
# ============================================================

def save_dataset(df, file_path):

    # Explicitly write as TSV
    df.to_csv(
        file_path,
        sep="\t",
        index=False,
        encoding="utf-8"
    )


# ============================================================
# SAVE NORMALIZATION CONFIGURATION
# ============================================================

def save_normalization_config(file_path):

    config = {

        "version": "1.0",

        "input_format": "TSV",

        "output_format": "TSV",

        "normalization_pipeline": [

            "Unicode NFKC normalization",

            "Lowercase conversion",

            "Whitespace normalization",

            "Punctuation normalization",

            "Company abbreviation expansion",

            "Legal suffix extraction",

            "Address abbreviation expansion",

            "Address number normalization",

            "Country canonicalization",

            "Tokenization"

        ],

        "company_abbreviations":
            COMPANY_ABBREVIATIONS,

        "address_abbreviations":
            ADDRESS_ABBREVIATIONS,

        "country_mapping":
            COUNTRY_MAPPING,

        "design_decisions": {

            "preserve_original_values": True,

            "preserve_legal_suffix": True,

            "aggressive_typo_correction": False,

            "phonetic_normalization": False,

            "transliteration": False,

            "original_file_format": "TSV",

            "output_file_format": "TSV"

        },

        "generated_columns": {

            "name_original":
                "Original company name",

            "address_original":
                "Original address",

            "country_original":
                "Original country",

            "name_normalized":
                "Fully normalized company name",

            "name_core":
                "Company name without recognized legal suffix",

            "name_legal_suffix":
                "Extracted legal suffix",

            "name_tokens":
                "Tokenized normalized company name",

            "address_normalized":
                "Fully normalized address",

            "address_tokens":
                "Tokenized normalized address",

            "country_normalized":
                "Canonical country code",

            "entity_normalization_key":
                "Combined normalized entity key"

        }

    }

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# COMPLETE NORMALIZATION PIPELINE
# ============================================================

def normalize_dataset(
    source,
    normalized_output
):

    print("=" * 70)
    print("ENTITY DATASET NORMALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load TSV
    # --------------------------------------------------------

    print("\n[1] Loading TSV dataset...")

    df = load_dataset(source)

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print(
        f"Columns  : {list(df.columns)}"
    )

    # --------------------------------------------------------
    # 2. Find required columns
    # --------------------------------------------------------

    print("\n[2] Validating required columns...")

    columns_lower = {
        column.lower(): column
        for column in df.columns
    }

    required_columns = [
        "business_name",
        "business_address",
        "country"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in columns_lower
    ]

    if missing_columns:

        raise ValueError(
            f"\nMissing required columns: "
            f"{missing_columns}\n\n"
            "Expected columns:\n"
            f"  {required_columns}"
            f"Available columns:\n{list(df.columns)}"
        )

    name_column = columns_lower[required_columns[0]]
    address_column = columns_lower[required_columns[1]]
    country_column = columns_lower[required_columns[2]]

    # --------------------------------------------------------
    # 3. Preserve originals
    # --------------------------------------------------------

    print("\n[3] Preserving original values...")

    df["name_original"] = df[name_column]

    df["address_original"] = df[address_column]

    df["country_original"] = df[country_column]

    # --------------------------------------------------------
    # 4. Normalize names
    # --------------------------------------------------------

    print("\n[4] Normalizing company names...")

    name_results = df[
        name_column
    ].apply(
        normalize_company_name
    )

    df["name_normalized"] = name_results.apply(
        lambda x: x["name_normalized"]
    )

    df["name_core"] = name_results.apply(
        lambda x: x["name_core"]
    )

    df["name_legal_suffix"] = name_results.apply(
        lambda x: x["name_legal_suffix"]
    )

    df["name_tokens"] = name_results.apply(
        lambda x: x["name_tokens"]
    )

    # --------------------------------------------------------
    # 5. Normalize addresses
    # --------------------------------------------------------

    print("\n[5] Normalizing addresses...")

    address_results = df[
        address_column
    ].apply(
        normalize_address
    )

    df["address_normalized"] = address_results.apply(
        lambda x: x["address_normalized"]
    )

    df["address_tokens"] = address_results.apply(
        lambda x: x["address_tokens"]
    )

    # --------------------------------------------------------
    # 6. Normalize country
    # --------------------------------------------------------

    print("\n[6] Normalizing countries...")

    df["country_normalized"] = df[
        country_column
    ].apply(
        normalize_country
    )

    # --------------------------------------------------------
    # 7. Create combined normalization key
    # --------------------------------------------------------

    print("\n[7] Creating normalization key...")

    df["entity_normalization_key"] = (
        df["name_core"].fillna("")
        + "|"
        + df["address_normalized"].fillna("")
        + "|"
        + df["country_normalized"].fillna("")
    )

    # --------------------------------------------------------
    # 8. Save TSV
    # --------------------------------------------------------

    print("\n[8] Saving normalized TSV...")

    save_dataset(
        df,
        normalized_output
    )

    # --------------------------------------------------------
    # 9. Save normalization configuration
    # --------------------------------------------------------

    print(
        "\n[9] Saving normalization configuration..."
    )

    save_normalization_config(
        normalization_config_file
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("NORMALIZATION COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print(
        f"\nInput TSV:"
        f"\n  {source}"
    )

    print(
        f"\nOutput TSV:"
        f"\n  {normalized_output}"
    )

    print(
        f"\nNormalization configuration:"
        f"\n  {normalization_config_file}"
    )

    print(
        f"\nFinal rows:"
        f"\n  {len(df):,}"
    )

    print(
        f"\nFinal columns:"
        f"\n  {len(df.columns)}"
    )

    print("\nGenerated columns:")

    generated_columns = [
        "name_original",
        "address_original",
        "country_original",
        "name_normalized",
        "name_core",
        "name_legal_suffix",
        "name_tokens",
        "address_normalized",
        "address_tokens",
        "country_normalized",
        "entity_normalization_key"
    ]

    for column in generated_columns:
        print(f"  + {column}")

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    assign_path(sys.argv[1], sys.argv[2])

    normalized_df = normalize_dataset(
        source,
        normalized_output
    )

    print("\nPreview:")
    print("-" * 70)

    preview_columns = [
        "name_original",
        "name_normalized",
        "name_core",
        "address_original",
        "address_normalized",
        "country_original",
        "country_normalized"
    ]

    print(
        normalized_df[
            preview_columns
        ].head(5).to_string(index=False)
    )
"""
```

The important part for your requirement is that the program uses:

```python
pd.read_csv(
    source,
    sep="\t",
    dtype=str,
    keep_default_na=False
)
```

and:

```python
df.to_csv(
    normalized_output,
    sep="\t",
    index=False,
    encoding="utf-8"
)
```

So the output remains a **proper TSV file**, not CSV.

### Output files

After execution:

```text
project/
│
├── source.tsv
│
├── normalized_output.tsv        ← normalized dataset
│
└── normalization_config.json    ← exact rules used
```

`normalization_config.json` is intentionally separate from the TSV because it contains the dictionaries and processing configuration. This will be useful when we build the **S2/S3 matching stage**, because we need to guarantee that every incoming dataset goes through the **same normalization process** before candidate generation and similarity scoring.
"""