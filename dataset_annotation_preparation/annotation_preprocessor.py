import os
import pandas as pd
import numpy as np
import random


def prepare_annotation(
    raw_annotation_path: str,
    images_dir: str,
    preprocessed_annotation_path: str,
    train_ratio: float = 0.8,
    pairing_type: str = "couples",
    n_augmentation: int = 1,
    n_error: int = 1,
):
    """
    Main function to prepare verification annotations.

    Args:
        raw_annotation_path: Path to the original Veri-Wild annotation CSV.
        images_dir: Directory containing images.
        preprocessed_annotation_path: Where to save the output CSV.
        train_ratio: Proportion of samples used for training.
        pairing_type: 'couples', 'triplets', or 'test'.
        n_augmentation: Number of pair shuffles to generate.
        n_error: Number of negative pairs/triplets to create.

    Returns:
        pd.DataFrame: Processed annotations.
    """
    check_pairing_type(pairing_type)

    df_annotation = pd.read_csv(raw_annotation_path, sep=";")

    df_processed = (
        df_annotation.pipe(process_object_info, image_directory=images_dir)
        .pipe(group_images_by_id)
        .pipe(create_image_pairs, n_augmentation=n_augmentation)
        .pipe(validate_correct_img_path)
        .pipe(
            apply_pairing_strategy,
            pairing_type=pairing_type,
            train_ratio=train_ratio,
            n_error=n_error,
        )
        .pipe(verif_couples, pairing_type=pairing_type)
    )

    df_processed.to_csv(preprocessed_annotation_path, index=False)

    return df_processed


def check_pairing_type(pairing_type: str):
    """
    Raises an error if pairing_type is not valid.
    """
    valid_types = {"triplets", "test", "couples"}

    if pairing_type not in valid_types:
        raise ValueError(
            f'Invalid pairing_type "{pairing_type}". Choose from {", ".join(valid_types)}.'
        )


def process_object_info(df, image_directory):

    df_object_info = df.copy()
    df_object_info[["id", "image"]] = df_object_info["id/image"].str.split(
        "/", expand=True
    )
    df_object_info["img_path"] = df_object_info["id/image"].apply(
        lambda path: os.path.join(image_directory, path) + ".jpg"
    )

    df_object_info["id"] = (
        df_object_info["id"] + "/" + df_object_info["Camera ID"].astype(str)
    )

    return df_object_info


def group_images_by_id(df_object_info):
    """
    Groups images by 'id_unique' and filters groups with more than one image.
    """
    grouped = df_object_info.groupby(["id"])["img_path"].apply(list)
    df_grouped = (
        grouped[grouped.apply(len) > 1]
        .reset_index()
        .rename(columns={"img_path": "img_list"})
    )
    return df_grouped


def create_image_pairs(df_grouped, n_augmentation=1):
    """
    Creates consecutive pairs of shuffled image paths and structures the DataFrame.
    """

    def create_consecutive_pairs(img_list):

        all_pairs = []
        for _ in range(n_augmentation):
            temp_list = img_list[:]
            random.shuffle(temp_list)
            if len(temp_list) % 2 == 1:
                temp_list = temp_list[:-1]
            all_pairs.extend(
                [(temp_list[i], temp_list[i + 1]) for i in range(0, len(temp_list), 2)]
            )
        return all_pairs

    df_grouped["couples"] = df_grouped["img_list"].apply(create_consecutive_pairs)
    df_couples = df_grouped.explode("couples").reset_index(drop=True)
    df_couples[["img_path", "couple_path"]] = pd.DataFrame(
        df_couples["couples"].tolist(), index=df_couples.index
    )

    return df_couples.drop(columns=["couples", "img_list"])


def validate_correct_img_path(df):

    mask_dir = (df["img_path"].apply(os.path.exists)) & (
        df["couple_path"].apply(os.path.exists)
    )

    df = df[mask_dir].reset_index(drop=True)

    return df


def filter_objects_by_camera_proportion(df):
    """
    Filters the objects based on the proportion of each Camera ID in the dataset.
    """
    threshold = 0.005

    # Total number of rows in the DataFrame
    total_rows = df.shape[0]

    # Calculate the proportion of each Camera ID
    camera_proportions = df.groupby("camera_id")["id"].count() / total_rows

    # Filter Camera IDs with a proportion greater than the threshold
    valid_cameras = camera_proportions[camera_proportions > threshold].index

    # Filter the DataFrame to include only the rows with valid Camera IDs
    filtered_df = df[df["camera_id"].isin(valid_cameras)].reset_index(drop=True)

    return filtered_df, valid_cameras


def create_couples(df, n_error=5):
    """
    Creates a new dataset by modifying and shuffling specific "couple" columns.
    """

    df_copy = df.copy()
    df_copy["camera_id"] = df_copy["id"].str.split("/").apply(lambda l: l[1])
    df_copy, valid_cameras = filter_objects_by_camera_proportion(df_copy)

    img_columns = df_copy.columns[df_copy.columns.str.contains("img")].to_list() + [
        "camera_id"
    ]
    df_img_col = df_copy[img_columns]
    df_img_col.columns = df_img_col.columns.str.replace("img_", "")

    couple_columns = df_copy.columns[
        df_copy.columns.str.contains("couple")
    ].to_list() + ["camera_id"]
    df_couple_col = df_copy[couple_columns]
    df_couple_col.columns = df_couple_col.columns.str.replace("couple_", "")

    names = ["img_", "couple_"]
    list_concat = []
    for camera_id in valid_cameras:
        mask = df_copy["camera_id"] == camera_id
        choice_df = [
            df_img_col[mask].reset_index(drop=True),
            df_couple_col[mask].reset_index(drop=True),
        ]

        for i in range(0, n_error + 1):
            choice = random.randint(0, 1)

            df_shuffled = choice_df[choice].copy().sample(frac=1).reset_index(drop=True)
            if i == 0:
                df_shuffled = choice_df[choice].copy()

            df_not_shuffled = choice_df[1 - choice].copy()

            df_shuffled.columns = names[choice] + df_shuffled.columns
            df_not_shuffled.columns = names[1 - choice] + df_not_shuffled.columns
            df_combined = pd.concat([df_shuffled, df_not_shuffled], axis=1)
            df_combined["label"] = 0

            if i == 0:
                df_combined["label"] = 1

            list_concat.append(df_combined.reset_index(drop=True))

    return pd.concat(list_concat).sample(frac=1).reset_index(drop=True)


def create_triplets(df):
    """
    Creates a new dataset by modifying and shuffling specific "couple" columns.
    """

    df_copy = df.copy()
    df_copy["camera_id"] = df_copy["id"].str.split("/").apply(lambda l: l[1])
    df_copy, valid_cameras = filter_objects_by_camera_proportion(df_copy)

    list_concat = []
    names = ["img_", "couple_"]
    for camera_id in valid_cameras:
        mask = df_copy["camera_id"] == camera_id
        df_camera = df_copy[mask].copy().reset_index(drop=True)

        choice = random.randint(0, 1)
        df_shuffled = (
            df_camera[["img_path", "couple_path"]]
            .copy()
            .sample(frac=1)
            .reset_index(drop=True)
        )

        df_camera["error_path"] = df_shuffled[names[choice] + "path"]
        list_concat.append(df_camera.reset_index(drop=True))

    return pd.concat(list_concat).sample(frac=1).reset_index(drop=True)


def apply_pairing_strategy(df, pairing_type, train_ratio, n_error=1):
    """
    Applies the correct pairing strategy based on the pairing_type.
    """
    strategies = {
        "couples": lambda df: df.pipe(create_couples, n_error=n_error).pipe(
            assign_train_column, train_ratio=train_ratio
        ),
        "test": lambda df: df.pipe(create_couples, n_error=n_error).pipe(
            assign_train_column, train_ratio=0.0
        ),
        "triplets": lambda df: df.pipe(create_triplets).pipe(
            assign_train_column, train_ratio=train_ratio
        ),
    }
    return strategies[pairing_type](df)


def assign_train_column(df, train_ratio):
    """
    Assign 1 for training and 0 for Testing
    """

    df["train"] = np.random.choice(
        [1, 0], size=len(df), p=[train_ratio, 1 - train_ratio]
    )

    return df


def verif_couples(df_annotation, pairing_type):
    def extract_id(filepath):
        # Handles NaNs or missing paths safely
        if not isinstance(filepath, str):
            return None
        id = os.path.basename(os.path.dirname(filepath))
        return id

    # Determine which image columns are relevant
    img_columns = ["img", "couple"]
    if pairing_type == "triplets":
        img_columns.append("error")

    # Extract IDs from filenames
    for col in img_columns:
        df_annotation[f"{col}_object_id"] = df_annotation[f"{col}_path"].apply(
            extract_id
        )

    if pairing_type == "triplets":
        # Check for invalid triplet logic
        mask = (
            (df_annotation["couple_object_id"] == df_annotation["error_id"])
            | (df_annotation["img_object_id"] == df_annotation["error_object_id"])
            | (df_annotation["img_object_id"] != df_annotation["couple_object_id"])
        )
    else:
        # For pairs: label 0 means different people, label 1 means same
        mask_error = df_annotation["label"] == 0
        mask_same_id = (
            df_annotation["couple_object_id"] == df_annotation["img_object_id"]
        )
        # Invalid if: label 0 but same ID, or label 1 but different ID
        mask = (mask_same_id & mask_error) | (~mask_same_id & ~mask_error)

    # Return only valid rows
    return df_annotation[~mask].reset_index(drop=True)
