import os
import pandas as pd
import numpy as np
import random
import json


def prepare_coco_annotation(
    raw_annotation_path: str,
    images_dir: str,
    preprocessed_annotation_path: str,
    train_ratio: float = 0.8,
    pairing_type: str = "couples",
    n_error: float = 1,
    n_augmentation: int = 1,
    error_data_path: str = None,
):
    """
    Processes raw annotations (COCO-style JSON) into object Verification dataset.

    Args:
        raw_annotation_path (str): Path to the JSON annotation file.
        images_dir (str): Directory containing the image files.
        preprocessed_annotation_path (str): Output path to save the processed CSV.
        train_ratio (float): Ratio of samples used for training (rest is test).
        pairing_type (str): Pairing strategy ("couples" or "test").
        n_error (float): Number of mismatched (negative) pairs per matched one.
        n_augmentation (int): Number of data augmentations per sample.
        error_data_path (str): Path to known error file (optional).

    Returns:
        pd.DataFrame: Final annotated and paired dataset.
    """
    with open(raw_annotation_path, "r", encoding="utf-8") as file:
        train_data = json.load(file)

    images = train_data["images"]
    annotations = train_data["annotations"]

    df_annotations = prepare_df_annotations(annotations)
    df_images = prepare_df_images(images)

    df_images_annotated = df_images.merge(df_annotations, on="image_id")

    df_processed = (
        df_images_annotated.pipe(create_path, data_directory=images_dir)
        .pipe(validate_correct_img_path)
        .pipe(filter_by_area_scale)
        .pipe(remove_least_frequent_views)
        .pipe(clean_bad_images, error_data_path=error_data_path)
        .pipe(
            apply_pairing_strategy,
            pairing_type=pairing_type,
            train_ratio=train_ratio,
            n_error=n_error,
            n_augmentation=n_augmentation,
        )
        .pipe(make_id_object)
        .pipe(verif_couples)
    )

    df_processed.to_csv(preprocessed_annotation_path, index=False)

    return df_processed


def prepare_df_annotations(annotations) -> pd.DataFrame:
    df_annotations = pd.DataFrame(annotations)

    # Keep only a specified category
    object_category = []

    df_annotations = df_annotations[
        df_annotations["category_id"].isin(object_category)
    ].reset_index(drop=True)
    df_annotations = df_annotations.rename(columns={"id": "annotation_id"})

    columns_drop_annotation = [
        "iscrowd",
        "category_id",
        "subcategory_id",
        "occlusion",
        "segmentation",
    ]
    df_annotations.drop(columns=columns_drop_annotation, inplace=True)

    return df_annotations


def prepare_df_images(images) -> pd.DataFrame:

    df_images = pd.DataFrame(images)
    df_images = df_images.rename(columns={"id": "image_id"})

    df_images = df_images[
        (df_images["acquisition_light"] != "unknown") & ~(df_images["location"].isna())
    ].reset_index(drop=True)

    # Create id

    df_images["id"] = (
        df_images["site"].astype(str)
        + "_"
        + df_images["config"].astype(str)
        + "_"
        + df_images["type"].astype(str)
    )

    columns_drop_images = [
        "location",
        "site",
        "config",
        "type",
    ]

    df_images.drop(columns=columns_drop_images, inplace=True)

    return df_images


def create_path(df: pd.DataFrame, data_directory: str) -> pd.DataFrame:

    df["file_name"] = df["file_name"].apply(
        lambda path: os.path.join(data_directory, path)
    )

    return df


def validate_correct_img_path(df: pd.DataFrame) -> pd.DataFrame:

    mask_dir = df["file_name"].apply(os.path.exists)
    df = df[mask_dir].reset_index(drop=True)
    return df


def filter_by_area_scale(
    df: pd.DataFrame, lower_quantile=0.85, upper_quantile=0.995
) -> pd.DataFrame:
    """
    Filters a DataFrame based on the relative size (area scale) of objects in images.

    """
    # Calculate area scale (relative size of the object in the image)
    df["area_scale"] = df["area"] / (df["width"] * df["height"])

    # Determine thresholds based on quantiles
    min_threshold = df["area_scale"].quantile(lower_quantile)
    max_threshold = df["area_scale"].quantile(upper_quantile)

    # Filter rows based on thresholds and return the cleaned DataFrame
    filtered_df = df[
        (df["area_scale"] >= min_threshold) & (df["area_scale"] <= max_threshold)
    ]

    # Reset index and drop the temporary "area_scale" column
    return filtered_df.reset_index(drop=True).drop(columns="area_scale")


def remove_least_frequent_views(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Removes the 'n' least frequent views based on the number of associated image files.
    """
    # Create a combined view identifier
    df["view_id"] = df["id"] + "_" + df["acquisition_light"]

    # Identify the 'n' least frequent views
    least_frequent_views = (
        df.groupby("view_id")["file_name"].count().sort_values().head(n).index.tolist()
    )

    # Filter out those least frequent views
    filtered_df = df[~df["view_id"].isin(least_frequent_views)].reset_index(drop=True)

    return filtered_df


def clean_bad_images(df: pd.DataFrame, error_data_path: str) -> pd.DataFrame:
    if not error_data_path:
        return df

    exclude_patterns = ""

    df = (
        df[~df["file_name"].str.contains(exclude_patterns, regex=True)]
        .reset_index(drop=True)
        .copy()
    )

    df_errors = pd.read_csv(error_data_path, sep=";")
    set_errors = set(df_errors[["path", "crop"]].apply(tuple, axis=1))

    df = df[
        ~df[["file_name", "bbox"]].apply(tuple, axis=1).isin(set_errors)
    ].reset_index(drop=True)

    return df


def create_couples(
    df_images_annotated: pd.DataFrame, n_error: int = 1, n_augmentation: int = 1
) -> pd.DataFrame:
    df_copy = df_images_annotated[["file_name", "bbox", "view_id"]].copy()
    df_copy = df_copy.rename(
        columns={
            "file_name": "path",
            "bbox": "crop",
            "view_id": "camera_id",
        }
    )
    df_copy = pd.concat([df_copy] * n_augmentation, ignore_index=True)
    df_path_col = df_copy.copy()
    df_couple_col = df_copy.copy()

    names = ["img_", "couple_"]
    list_concat = []

    valid_cameras = df_copy["camera_id"].unique()

    for camera_id in valid_cameras:
        mask = df_copy["camera_id"] == camera_id
        choice_df = [
            df_path_col[mask].reset_index(drop=True),
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


def apply_pairing_strategy(
    df: pd.DataFrame,
    pairing_type: str,
    train_ratio: float,
    n_error: int = 1,
    n_augmentation: int = 1,
) -> pd.DataFrame:
    """
    Applies the correct pairing strategy based on the pairing_type.
    """
    strategies = {
        "couples": lambda df: df.pipe(
            create_couples, n_error=n_error, n_augmentation=n_augmentation
        ).pipe(assign_train_column, train_ratio=train_ratio),
        "test": lambda df: df.pipe(
            create_couples, n_error=n_error, n_augmentation=n_augmentation
        ).pipe(assign_train_column, train_ratio=0.0),
    }
    return strategies[pairing_type](df)


def assign_train_column(df: pd.DataFrame, train_ratio: float) -> pd.DataFrame:
    """
    Assign 1 for training and 0 for Testing
    """

    df["train"] = np.random.choice(
        [1, 0], size=len(df), p=[train_ratio, 1 - train_ratio]
    )

    return df


def verif_couples(df_annotation: pd.DataFrame) -> pd.DataFrame:

    mask_error = df_annotation["label"] == 0
    mask_same_object = (
        df_annotation["img_object_id"] == df_annotation["couple_object_id"]
    )

    mask = mask_error & mask_same_object
    return df_annotation[~mask].reset_index(drop=True)


def make_id_object(df: pd.DataFrame) -> pd.DataFrame:

    for col_type in ["img_", "couple_"]:
        df[col_type + "object_id"] = (
            df[col_type + "path"].apply(os.path.dirname).apply(os.path.basename)
        )

    return df
