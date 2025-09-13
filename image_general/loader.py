import torch
import pandas as pd
from utils import image_utils


class ImagePreprocessor:
    """
    ImagePreprocessor handles preprocessing of annotated images, including optional
    anonymization and cropping. It supports various transformations, model inference,
    and flexible loading from different data sources (e.g., object or camera).

    Attributes:
        img_annotations (pd.DataFrame): Parsed annotation data from CSV.
        file_type (str): Image file extension (e.g., 'png', 'jpg').
        crop_type (str): Type of object to crop ('object', 'person', etc.).
        anonymize_plate (bool): Whether to anonymize license plates in images.
        transform (callable): Transformations to apply to the images.
        algo_pair (str): Image processing algorithm combination identifier.
        object_dir (str): Path to object data, if any.
        data_dir (str): Path to image data.
        device (str): Device to run the model on ('cpu' or 'cuda').
        model (torch.nn.Module): Optional model for additional processing.
    """

    def __init__(
        self,
        raw_annotation_path: str,
        transform,
        object_dir=None,
        data_dir=None,
        file_type="png",
        crop_type="object",
        anonymize_plate=False,
        algo_pair="NoBayer:GrabEq",
        device="cpu",
        model=None,
    ):
        self.img_annotations = pd.read_csv(raw_annotation_path)
        self.file_type = file_type
        self.crop_type = crop_type
        self.anonymize_plate = anonymize_plate
        self.transform = transform
        self.algo_pair = algo_pair

        self.object_dir = object_dir
        self.data_dir = data_dir
        self.device = device
        self.model = model

        if model:
            self.model.to(device)
            self.model.eval()

    def get_rows_img(self, ids, type_id):
        """
        Retrieves annotation rows by ID or index.

        Args:
            ids (list or int): List of IDs or indices to retrieve.
            type_id (str): Either 'index' for row index lookup or column name (e.g., 'image_id').

        Returns:
            list or pd.Series: Retrieved annotation data. If 'index', returns a Series.
                               Otherwise, returns a list of row dictionaries.
        """

        if type_id == "index":
            return self.img_annotations.loc[ids]
        return self.img_annotations[self.img_annotations[type_id].isin(ids)].to_dict(
            orient="records"
        )
