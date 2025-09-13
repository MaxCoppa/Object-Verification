# %%
import os
import pandas as pd
from utils import image_utils
from veri_models import ObjectVeriSiamese, ModelEnsembler
import torch
import random

from image_general import ImagePreprocessor, ImageFactory


# %%

file_type = "png"
crop_type = "object"
device = "cuda"


transform_default = image_utils.transform_fc("test")

best_model_path = "pretrained_model/object_verif_eval.pt"
model = torch.load(best_model_path, weights_only=False)
# %%
preprocessor = ImagePreprocessor(
    raw_annotation_path="annotations/annotations_cleaned_pred_demo.csv",
    transform=transform_default,
    file_type=file_type,
    crop_type=crop_type,
    anonymize_plate=True,
    model=model,
    device=device,
)

img_builder = ImageFactory(preprocessor)
plates_all = preprocessor.img_annotations["plate"].unique()

# %%


def demo_prediction(label=1):
    example = random.choice(plates_all)
    error = random.choice(plates_all)

    rows_example = preprocessor.get_rows_img([example], "plate")

    rows_error = preprocessor.get_rows_img([error], "plate")
    if label:
        img_obj = img_builder.create_img(row_img=rows_example[0])
        couple_obj = img_builder.create_img(row_img=rows_example[1])
        couple = img_builder.create_couple(img1=img_obj, img2=couple_obj)

    else:
        img_obj = img_builder.create_img(row_img=rows_example[0])
        couple_obj = img_builder.create_img(row_img=rows_error[0])
        couple = img_builder.create_couple(img1=img_obj, img2=couple_obj)

    if couple.check_valid_hours():
        couple.predict()
        couple.show()

    else:
        demo_prediction(label)


# %%
label = 1
demo_prediction(label=label)

# %%
