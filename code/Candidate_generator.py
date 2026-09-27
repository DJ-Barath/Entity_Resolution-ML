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

    value = record.get(column, "")

    if value is None:
        return ""

    return str(value).strip()


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

    if embeddings.size == 0:
        return embeddings

    faiss.normalize_L2(embeddings)

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
    Build an HNSW FAISS index using cosine similarity.

    Embeddings must already be L2 normalized.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional array."
        )

    dimension = embeddings.shape[1]

    # Explicitly use Inner Product.
    #
    # Since embeddings are L2-normalized:
    #
    #     Inner Product == Cosine Similarity
    #
    index = faiss.IndexHNSWFlat(
        dimension,
        config.M
    )

    index.metric_type = config.METRIC

    index.hnsw.efConstruction = (
        config.EF_CONSTRUCTION
    )

    index.hnsw.efSearch = (
        config.EF_SEARCH
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# Feature Index Construction
# ============================================================

def build_feature_indexes(
    s1_records,
    model
):
    """
    Build independent FAISS indexes for:

        name
        address
        country

    Also retain the normalized S1 embeddings.

    Returns:

        {
            "name": {
                "index": ...,
                "embeddings": ...
            },

            "address": {
                "index": ...,
                "embeddings": ...
            },

            "country": {
                "index": ...,
                "embeddings": ...
            }
        }
    """

    feature_columns = {
        "name": config.NAME_COLUMN,
        "address": config.ADDRESS_COLUMN,
        "country": config.COUNTRY_COLUMN,
    }

    feature_indexes = {}

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
            f"Building {feature_name} HNSW index..."
        )

        index = build_index(
            embeddings
        )

        feature_indexes[
            feature_name
        ] = {
            "index": index,
            "embeddings": embeddings
        }

        print(
            f"{feature_name.capitalize()} index "
            f"records: {len(s1_records):,}"
        )

        print(
            f"{feature_name.capitalize()} "
            f"embedding dimension: "
            f"{embeddings.shape[1]}"
        )

        print(
            f"{feature_name.capitalize()} "
            f"HNSW M: {config.M}"
        )

        print(
            f"{feature_name.capitalize()} "
            f"efConstruction: "
            f"{config.EF_CONSTRUCTION}"
        )

        print(
            f"{feature_name.capitalize()} "
            f"efSearch: "
            f"{config.EF_SEARCH}"
        )

    return feature_indexes


# ============================================================
# Search Feature
# ============================================================

def search_feature(
    source_embeddings,
    index,
    top_k
):
    """
    Search a feature-specific HNSW index.

    source_embeddings must already be normalized.
    """

    similarities, indices = index.search(
        source_embeddings,
        top_k
    )

    return (
        similarities,
        indices
    )


# ============================================================
# Candidate Pool Generation
# ============================================================

def generate_candidate_pool(
    source_embeddings,
    feature_indexes
):
    """
    Generate a UNION of candidates retrieved independently
    from name, address and country HNSW indexes.

    Important:
        This function ONLY performs candidate retrieval.

        It does NOT calculate the final weighted score.
    """

    feature_top_k = {
        "name": config.NAME_TOP_K,
        "address": config.ADDRESS_TOP_K,
        "country": config.COUNTRY_TOP_K,
    }

    feature_results = {}

    for feature_name, top_k in feature_top_k.items():

        print(
            f"\nSearching {feature_name} "
            f"TOP_K={top_k}..."
        )

        similarities, indices = search_feature(
            source_embeddings=source_embeddings[
                feature_name
            ],
            index=feature_indexes[
                feature_name
            ]["index"],
            top_k=top_k
        )

        feature_results[
            feature_name
        ] = {
            "similarities": similarities,
            "indices": indices
        }

    return feature_results


# ============================================================
# Actual Candidate Similarity
# ============================================================

def calculate_candidate_similarity(
    source_index,
    s1_index,
    source_embeddings,
    s1_embeddings
):
    """
    Calculate the actual cosine similarity between one
    source record and one S1 record.

    Both vectors are already L2-normalized.

    Therefore:

        dot(source, s1)
            =
        cosine similarity
    """

    similarity = np.dot(
        source_embeddings[source_index],
        s1_embeddings[s1_index]
    )

    return float(similarity)


# ============================================================
# Candidate Generation
# ============================================================

def generate_candidates(
    source_records,
    s1_records,
    feature_indexes,
    model
):
    """
    Generate candidate pairs.

    Pipeline:

        1. Encode source features.
        2. Retrieve candidates using HNSW.
        3. UNION candidates from all features.
        4. Calculate actual name similarity.
        5. Calculate actual address similarity.
        6. Calculate actual country similarity.
        7. Calculate weighted similarity.
        8. Apply optional threshold.
        9. Keep FINAL_TOP_K.

    Unlike the previous implementation, a feature that did
    not retrieve a candidate does NOT receive similarity 0.

    The actual similarity is calculated for every candidate
    in the union.
    """

    feature_columns = {
        "name": config.NAME_COLUMN,
        "address": config.ADDRESS_COLUMN,
        "country": config.COUNTRY_COLUMN,
    }

    # --------------------------------------------------------
    # Encode source features ONCE
    # --------------------------------------------------------

    source_embeddings = {}

    for feature_name, column in feature_columns.items():

        print()
        print(
            f"Encoding source {feature_name}..."
        )

        source_embeddings[
            feature_name
        ] = encode_feature(
            records=source_records,
            column=column,
            model=model
        )

    # --------------------------------------------------------
    # Generate HNSW candidate pool
    # --------------------------------------------------------

    feature_results = generate_candidate_pool(
        source_embeddings=source_embeddings,
        feature_indexes=feature_indexes
    )

    candidate_pairs = []

    # --------------------------------------------------------
    # Process each source record
    # --------------------------------------------------------

    for source_index, source_record in enumerate(
        source_records
    ):

        # ----------------------------------------------------
        # UNION of candidates from all feature indexes
        # ----------------------------------------------------

        candidate_indices = set()

        for feature_name in feature_columns:

            indices = feature_results[
                feature_name
            ]["indices"][source_index]

            for s1_index in indices:

                if s1_index < 0:
                    continue

                candidate_indices.add(
                    int(s1_index)
                )

        # ----------------------------------------------------
        # Calculate ACTUAL similarity for every candidate
        # ----------------------------------------------------

        print(
            f"Candidate Indices {source_record} for ",
            candidate_indices
        )
        ranked_candidates = []

        for s1_index in candidate_indices:
            # print(s1_index)
            # ------------------------------------------------
            # Actual name similarity
            # ------------------------------------------------

            name_similarity = (
                calculate_candidate_similarity(
                    source_index=source_index,
                    s1_index=s1_index,
                    source_embeddings=source_embeddings[
                        "name"
                    ],
                    s1_embeddings=feature_indexes[
                        "name"
                    ]["embeddings"]
                )
            )

            # ------------------------------------------------
            # Actual address similarity
            # ------------------------------------------------

            address_similarity = (
                calculate_candidate_similarity(
                    source_index=source_index,
                    s1_index=s1_index,
                    source_embeddings=source_embeddings[
                        "address"
                    ],
                    s1_embeddings=feature_indexes[
                        "address"
                    ]["embeddings"]
                )
            )

            # ------------------------------------------------
            # Actual country similarity
            # ------------------------------------------------

            country_similarity = (
                calculate_candidate_similarity(
                    source_index=source_index,
                    s1_index=s1_index,
                    source_embeddings=source_embeddings[
                        "country"
                    ],
                    s1_embeddings=feature_indexes[
                        "country"
                    ]["embeddings"]
                )
            )

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
            # print("combined_similarity : " + combined_similarity)
            if (
                config.SIMILARITY_THRESHOLD is not None
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
        # Rank candidates
        # ----------------------------------------------------

        ranked_candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        print(
            f" final RankedCandidates pair for {ranked_candidates}"
        )
        # ----------------------------------------------------
        # Keep FINAL_TOP_K
        # ----------------------------------------------------

        for rank, (
            combined_similarity,
            s1_index,
            name_similarity,
            address_similarity,
            country_similarity
        ) in enumerate(
            ranked_candidates[
                :config.FINAL_TOP_K
            ],
            start=1
        ):

            s1_record = s1_records[
                s1_index
            ]

            candidate_pair = {

                # --------------------------------------------
                # IDs
                # --------------------------------------------

                config.candidatePairs[
                    config.S1_INDEX
                ]: s1_record.get(
                    config.ENTITY_ID_COLUMN,
                    ""
                ),

                config.candidatePairs[
                    config.MATCH_INDEX
                ]: source_record.get(
                    config.ENTITY_ID_COLUMN,
                    ""
                ),

                # --------------------------------------------
                # Name
                # --------------------------------------------

                config.candidatePairs[
                    config.S1_NAME
                ]: s1_record.get(
                    config.NAME_COLUMN,
                    ""
                ),

                config.candidatePairs[
                    config.MATCH_NAME
                ]: source_record.get(
                    config.NAME_COLUMN,
                    ""
                ),

                # --------------------------------------------
                # Address
                # --------------------------------------------

                config.candidatePairs[
                    config.S1_ADDRESS
                ]: s1_record.get(
                    config.ADDRESS_COLUMN,
                    ""
                ),

                config.candidatePairs[
                    config.MATCH_ADDRESS
                ]: source_record.get(
                    config.ADDRESS_COLUMN,
                    ""
                ),

                # --------------------------------------------
                # Country
                # --------------------------------------------

                config.candidatePairs[
                    config.S1_COUNTRY
                ]: s1_record.get(
                    config.COUNTRY_COLUMN,
                    ""
                ),

                config.candidatePairs[
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

                config.candidatePairs[
                    config.SIMILARITY
                ]: combined_similarity,

                # --------------------------------------------
                # Rank
                # --------------------------------------------

                config.candidatePairs[
                    config.RANK
                ]: rank,
            }

            candidate_pairs.append(
                candidate_pair
            )

    print(
        f" final candidates pair for {candidate_pairs}"
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
    # Build separate feature indexes
    # --------------------------------------------------------

    print(
        "\nBuilding feature-specific HNSW indexes..."
    )

    feature_indexes = build_feature_indexes(
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
            feature_indexes=feature_indexes,
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