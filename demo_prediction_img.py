# %% -------------------------
# Import necessary libraries
# -------------------------
import random

# Project modules
from utils import image_utils
from veri_models import ObjectVeriSiamese
from image_general import ImagePreprocessor, ImageFactory

# Import project configuration
from config import ANNOTATIONS_PATH

# %% -------------------------
# Model and parameters
# -------------------------
model = ObjectVeriSiamese()

file_type = "jpg"
device = "cpu"

transform_default = image_utils.transform_fc("test")

# %% -------------------------
# Image preprocessor setup
# -------------------------
preprocessor = ImagePreprocessor(
    raw_annotation_path=ANNOTATIONS_PATH / "demo_annotations_sharks.csv",
    transform=transform_default,
    file_type=file_type,
    crop_type=None,
    model=model,
    device=device,
)

img_builder = ImageFactory(preprocessor)
ids_all = preprocessor.img_annotations["object_id"].unique()


# %% -------------------------
# Demo prediction function
# -------------------------
def demo_prediction(label: int = 1):
    """
    Generate a demo prediction using the Siamese network.

    Args:
        label (int): 1 for positive pair (same object),
                     0 for negative pair (different objects).
    """
    # Select one object ID
    example = random.choice(ids_all)

    # Select a different object ID
    error = random.choice(ids_all[ids_all != example])

    # Fetch image rows for both
    rows_example = preprocessor.get_rows_img([example], "object_id")
    random.shuffle(rows_example)

    rows_error = preprocessor.get_rows_img([error], "object_id")
    random.shuffle(rows_error)

    # Build either a positive or negative pair
    if label:
        img_obj = img_builder.create_img(row_img=rows_example[0])
        couple_obj = img_builder.create_img(row_img=rows_example[1])
        couple = img_builder.create_couple(img1=img_obj, img2=couple_obj)
    else:
        img_obj = img_builder.create_img(row_img=rows_example[0])
        couple_obj = img_builder.create_img(row_img=rows_error[0])
        couple = img_builder.create_couple(img1=img_obj, img2=couple_obj)

    # Run prediction and visualize
    couple.predict()
    couple.show()


# %% -------------------------
# Run demo
# -------------------------
label = 1  # Change to 0 for negative pair
demo_prediction(label=label)

# %%
