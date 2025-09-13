# %%
import os
import pandas as pd
from utils import image_utils
from veri_models import ObjectVeriSiamese, ModelEnsembler
import torch
import random

from image_general import ImagePreprocessor, ImageFactory

# %%
model = ObjectVeriSiamese()
# %%
# model
# %%

file_type = "jpg"
device = "cpu"


transform_default = image_utils.transform_fc("test")

# best_model_path = "pretrained_model/object_verif_eval.pt"
# model = torch.load(best_model_path, weights_only=False)
# %%
preprocessor = ImagePreprocessor(
    raw_annotation_path="/Users/maximecoppa/Desktop/Projects/Object-Verification/data/annotations/demo_annotations_sharks.csv",
    transform=transform_default,
    file_type=file_type,
    crop_type=None,
    model=model,
    device=device,
)

img_builder = ImageFactory(preprocessor)
ids_all = preprocessor.img_annotations["object_id"].unique()

# %%


def demo_prediction(label=1):
    example = random.choice(ids_all)
    error = random.choice(ids_all[ids_all != example])

    rows_example = preprocessor.get_rows_img([example], "object_id")
    random.shuffle(rows_example)

    rows_error = preprocessor.get_rows_img([error], "object_id")
    random.shuffle(rows_error)

    if label:
        img_obj = img_builder.create_img(row_img=rows_example[0])
        couple_obj = img_builder.create_img(row_img=rows_example[1])
        couple = img_builder.create_couple(img1=img_obj, img2=couple_obj)

    else:
        img_obj = img_builder.create_img(row_img=rows_example[0])
        couple_obj = img_builder.create_img(row_img=rows_error[0])
        couple = img_builder.create_couple(img1=img_obj, img2=couple_obj)

    couple.predict()
    couple.show()


# %%
label = 1
demo_prediction(label=label)


# %%
