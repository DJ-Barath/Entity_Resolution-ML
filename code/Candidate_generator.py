import csv
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import CandidateConfig as config


# ============================================================
# Utility Functions
# ============================================================

def load_tsv(path):
    """
    Load a TSV dataset.

    Returns:
        list[dict]
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        return list(reader)


def get_feature_text(record, column):
    """
    Get a single feature from a record.

    The feature is converted to a stripped string.
    """

    return record.get(
        column,
        ""
    ).strip()


def normalize_embeddings(embeddings):
    """
    L2-normalize embeddings.

    After normalization, inner product is equivalent
    to cosine similarity.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(
        embeddings
    )

    return embeddings


# ============================================================
# Feature Encoding
# ============================================================

def encode_feature(records, column, model):
    """
    Encode one feature independently.

    Example:

        name    -> name embeddings
        address -> address embeddings
        country -> country embeddings
    """

    texts = [
        get_feature_text(
            record,
            column
        )
        for record in records
    ]

    embeddings = model.encode(
        texts,
        batch_size=config.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return normalize_embeddings(
        embeddings
    )


# ============================================================
# FAISS Index
# ============================================================

def build_index(embeddings):
    """
    Build a FAISS cosine-similarity index
    from normalized embeddings.
    """

    dimension = embeddings.shape[1]

    # Flat indexing
    index = faiss.IndexFlatIP(
        dimension
    )

    # this is HNSW embedding
    # faiss.normalize_L2(embeddings)

    # index = faiss.IndexHNSWFlat(
    #     dimension, config.M
    # )
    
    # index.metric_type = faiss.METRIC_INNER_PRODUCT

    index.add(
        embeddings
    )

    return index


def build_feature_indexes(
    s1_records,
    model
):
    """
    Build independent FAISS indexes for:

        name
        address
        country

    Returns:
        {
            "name": name_index,
            "address": address_index,
            "country": country_index
        }
    """

    feature_columns = {
        "name": config.NAME_COLUMN,
        "address": config.ADDRESS_COLUMN,
        "country": config.COUNTRY_COLUMN,
    }

    indexes = {}

    for feature_name, column in feature_columns.items():

        print()
        print(
            f"Creating {feature_name} embeddings..."
        )

        embeddings = encode_feature(
            records=s1_records,
            column=column,
            model=model
        )

        print(
            f"Building {feature_name} FAISS index..."
        )

        indexes[
            feature_name
        ] = build_index(
            embeddings
        )

        print(
            f"{feature_name.capitalize()} index "
            f"records: {len(s1_records):,}"
        )

        print(
            f"{feature_name.capitalize()} "
            f"embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    return indexes


# ============================================================
# Search Feature
# ============================================================

def search_feature(
    source_records,
    column,
    index,
    model
):
    """
    Encode a source feature and search its corresponding
    FAISS index.

    Returns:
        similarities, indices
    """

    embeddings = encode_feature(
        records=source_records,
        column=column,
        model=model
    )

    # Normalization is required for HNSW embedding
    # faiss.normalize_L2(embeddings)

    similarities, indices = index.search(
        embeddings,
        config.TOP_K
    )

    return (
        similarities,
        indices
    )


# ============================================================
# Candidate Generation
# ============================================================

def generate_candidates(
    source_records,
    s1_records,
    indexes,
    model
):
    """
    Generate candidate pairs using separate FAISS
    indexes for name, address and country.

    Candidates from all feature indexes are combined
    using the union of their candidate IDs.

    Each candidate contains:

        name similarity
        address similarity
        country similarity
        combined similarity
    """

    # --------------------------------------------------------
    # Feature configuration
    # --------------------------------------------------------

    feature_columns = {
        "name": config.NAME_COLUMN,
        "address": config.ADDRESS_COLUMN,
        "country": config.COUNTRY_COLUMN,
    }

    # --------------------------------------------------------
    # Search each feature independently
    # --------------------------------------------------------

    feature_results = {}

    for feature_name, column in feature_columns.items():

        print(
            f"\nSearching {feature_name}..."
        )

        similarities, indices = search_feature(
            source_records=source_records,
            column=column,
            index=indexes[feature_name],
            model=model
        )

        feature_results[
            feature_name
        ] = {
            "similarities": similarities,
            "indices": indices
        }

    # --------------------------------------------------------
    # Combine candidates
    # --------------------------------------------------------

    CandidateKeys = config.candidatePairs

    candidate_pairs = []

    for source_index, source_record in enumerate(
        source_records
    ):

        # ----------------------------------------------------
        # Candidate map
        #
        # key:
        #     S1 record index
        #
        # value:
        #     feature similarities
        # ----------------------------------------------------

        candidates = {}

        for feature_name in feature_columns:

            similarities = feature_results[
                feature_name
            ][
                "similarities"
            ][source_index]

            indices = feature_results[
                feature_name
            ][
                "indices"
            ][source_index]

            for rank in range(
                config.TOP_K
            ):

                s1_index = indices[rank]

                similarity = similarities[rank]

                if s1_index < 0:
                    continue

                # ------------------------------------------------
                # Create candidate entry if it does not exist
                # ------------------------------------------------

                if s1_index not in candidates:

                    candidates[
                        s1_index
                    ] = {
                        "name_similarity": 0.0,
                        "address_similarity": 0.0,
                        "country_similarity": 0.0,
                    }

                # ------------------------------------------------
                # Store this feature's similarity
                # ------------------------------------------------

                candidates[
                    s1_index
                ][
                    f"{feature_name}_similarity"
                ] = float(
                    similarity
                )

        # ----------------------------------------------------
        # Calculate combined similarity
        # ----------------------------------------------------

        ranked_candidates = []

        for s1_index, scores in candidates.items():

            name_similarity = scores[
                "name_similarity"
            ]

            address_similarity = scores[
                "address_similarity"
            ]

            country_similarity = scores[
                "country_similarity"
            ]

            # ------------------------------------------------
            # Weighted similarity
            # ------------------------------------------------

            combined_similarity = (
                config.NAME_WEIGHT
                * name_similarity
                +
                config.ADDRESS_WEIGHT
                * address_similarity
                +
                config.COUNTRY_WEIGHT
                * country_similarity
            )

            # ------------------------------------------------
            # Optional threshold
            # ------------------------------------------------

            if (
                config.SIMILARITY_THRESHOLD
                is not None
                and
                combined_similarity
                < config.SIMILARITY_THRESHOLD
            ):
                continue

            ranked_candidates.append(
                (
                    combined_similarity,
                    s1_index,
                    name_similarity,
                    address_similarity,
                    country_similarity
                )
            )

        # ----------------------------------------------------
        # Rank candidates by combined similarity
        # ----------------------------------------------------

        ranked_candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # ----------------------------------------------------
        # Keep final TOP_K candidates
        # ----------------------------------------------------

        for rank, (
            combined_similarity,
            s1_index,
            name_similarity,
            address_similarity,
            country_similarity
        ) in enumerate(
            ranked_candidates[
                :config.TOP_K
            ],
            start=1
        ):

            s1_record = s1_records[
                s1_index
            ]

            # ------------------------------------------------
            # Create combined match pair
            # ------------------------------------------------

            candidate_pair = {

                # --------------------------------------------
                # IDs
                # --------------------------------------------

                CandidateKeys[
                    config.S1_INDEX
                ]: s1_record.get(
                    config.ENTITY_ID_COLUMN,
                    ""
                ),

                CandidateKeys[
                    config.MATCH_INDEX
                ]: source_record.get(
                    config.ENTITY_ID_COLUMN,
                    ""
                ),

                # --------------------------------------------
                # Name
                # --------------------------------------------

                CandidateKeys[
                    config.S1_NAME
                ]: s1_record.get(
                    config.NAME_COLUMN,
                    ""
                ),

                CandidateKeys[
                    config.MATCH_NAME
                ]: source_record.get(
                    config.NAME_COLUMN,
                    ""
                ),

                # --------------------------------------------
                # Address
                # --------------------------------------------

                CandidateKeys[
                    config.S1_ADDRESS
                ]: s1_record.get(
                    config.ADDRESS_COLUMN,
                    ""
                ),

                CandidateKeys[
                    config.MATCH_ADDRESS
                ]: source_record.get(
                    config.ADDRESS_COLUMN,
                    ""
                ),

                # --------------------------------------------
                # Country
                # --------------------------------------------

                CandidateKeys[
                    config.S1_COUNTRY
                ]: s1_record.get(
                    config.COUNTRY_COLUMN,
                    ""
                ),

                CandidateKeys[
                    config.MATCH_COUNTRY
                ]: source_record.get(
                    config.COUNTRY_COLUMN,
                    ""
                ),

                # --------------------------------------------
                # Individual similarities
                # --------------------------------------------

                "name_similarity":
                    name_similarity,

                "address_similarity":
                    address_similarity,

                "country_similarity":
                    country_similarity,

                # --------------------------------------------
                # Combined similarity
                # --------------------------------------------

                CandidateKeys[
                    config.SIMILARITY
                ]: combined_similarity,

                CandidateKeys[
                    config.RANK
                ]: rank,
            }

            candidate_pairs.append(
                candidate_pair
            )

    return candidate_pairs


# ============================================================
# Save Candidates
# ============================================================

def saveCandidates(
    candidate_pairs,
    output_path,
    columns_to_save
):
    """
    Save candidate pairs to a TSV file.

    Only columns specified in columns_to_save
    will be written.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not candidate_pairs:

        print(
            "No candidate pairs generated."
        )

        return

    # --------------------------------------------------------
    # Validate requested columns
    # --------------------------------------------------------

    available_columns = set(
        candidate_pairs[0].keys()
    )

    invalid_columns = [
        column
        for column in columns_to_save
        if column not in available_columns
    ]

    if invalid_columns:

        raise ValueError(
            "The following columns do not exist "
            f"in candidate pairs: {invalid_columns}"
        )

    # --------------------------------------------------------
    # Write selected columns
    # --------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns_to_save,
            delimiter="\t"
        )

        writer.writeheader()

        for candidate in candidate_pairs:

            writer.writerow({
                column: candidate.get(
                    column,
                    ""
                )
                for column in columns_to_save
            })

    print(
        f"Candidate pairs saved to: "
        f"{output_path}"
    )

    print(
        f"Rows written: "
        f"{len(candidate_pairs):,}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("ENTITY CANDIDATE BLOCKING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    print("\nLoading S1...")

    s1_records = load_tsv(
        config.S1_PATH
    )

    print(
        f"S1 records: "
        f"{len(s1_records):,}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        config.MODEL_NAME
    )

    # --------------------------------------------------------
    # Build separate FAISS indexes
    # --------------------------------------------------------

    print(
        "\nBuilding feature-specific indexes..."
    )

    indexes = build_feature_indexes(
        s1_records=s1_records,
        model=model
    )

    # --------------------------------------------------------
    # Process S2 and S3
    # --------------------------------------------------------

    all_candidates = []

    for source_path in (
        config.SOURCE_FILES.values()
    ):

        print()
        print("=" * 60)

        print(
            f"Loading dataset: "
            f"{source_path}"
        )

        print("=" * 60)

        source_records = load_tsv(
            source_path
        )

        print(
            f"Records: "
            f"{len(source_records):,}"
        )

        # ----------------------------------------------------
        # Generate candidates
        # ----------------------------------------------------

        candidates = generate_candidates(
            source_records=source_records,
            s1_records=s1_records,
            indexes=indexes,
            model=model
        )

        all_candidates.extend(
            candidates
        )

        print(
            f"Candidates generated: "
            f"{len(candidates):,}"
        )

    # --------------------------------------------------------
    # Output columns
    # --------------------------------------------------------

    columns_to_save = list(
        config.candidatePairsSave
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    saveCandidates(
        candidate_pairs=all_candidates,
        output_path=config.OUTPUT_PATH,
        columns_to_save=columns_to_save
    )

    print(
        "\nCandidate blocking completed."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()