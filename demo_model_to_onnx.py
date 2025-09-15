# %% -------------------------
# Import necessary libraries
# -------------------------
import torch
import time
from utils import image_utils, onnx_utils
from veri_models import ObjectVeriSiameseMC, ModelEnsemblerMC
import random as rd
from tqdm import tqdm

from image_general import ImagePreprocessor, ImageFactory


# Project configuration paths
from config import ANNOTATIONS_PATH, RAW_ANNOTATIONS_PATH, DATA_PATH, MODELS_PATH

# %% -------------------------
# Load PyTorch model
# -------------------------
start_load_model = time.time()  # start timing model load

# Initialize Siamese model
model = ObjectVeriSiameseMC()

# Load pretrained model weights
model.load_state_dict(torch.load(MODELS_PATH / "model_1.pth"))

# Optional: multiple model ensemble (commented out)
# model1 = ObjectVeriSiameseMC(backbone="efficientnet_v2_s")
# model1.load_state_dict(torch.load("pretrained_model/comparaison/comparaison_67.pth"))
# model2 = ObjectVeriSiameseMC()
# model2.load_state_dict(torch.load("pretrained_model/comparaison/comparaison_101.pth"))
# model3 = ObjectVeriSiameseMC()
# model3.load_state_dict(torch.load("pretrained_model/comparaison/comparaison_100.pth"))
# model = ModelEnsemblerMC(models=[model1, model2, model3])

model.to("cpu")  # move model to CPU
model.eval()  # set model to evaluation mode
end_load_model = time.time()  # end timing
print(f"Model loading time: {end_load_model - start_load_model:.2f}s")

# %% -------------------------
# Export PyTorch model to ONNX
# -------------------------
example_inputs = (
    torch.randn(1, 3, 224, 224),
    torch.randn(1, 3, 224, 224),
)  # dummy input tensors
onnx_model_path = MODELS_PATH / "model_1.onnx"  # path to save ONNX model

# Export model to ONNX format
is_export = onnx_utils.export_model_onnx(
    model=model, example_inputs=example_inputs, onnx_model_path=onnx_model_path
)

# %% -------------------------
# Load ONNX model and create inference session
# -------------------------
start_onnx_load = time.time()

onnx_model = onnx_utils.load_model_onnx(
    onnx_model_path=onnx_model_path
)  # load ONNX model
ort_session = onnx_utils.create_ort_session(
    onnx_model_path
)  # create ONNX Runtime session

end_onnx_load = time.time()
print(
    f"ONNX model load + session creation time: {end_onnx_load - start_onnx_load:.2f}s"
)

# %% -------------------------
# Image preprocessor setup
# -------------------------
file_type = "jpg"  # image file type
device = "cpu"  # device for processing

transform_default = image_utils.transform_fc("test")  # default image transform

# Initialize preprocessor with annotations and model
preprocessor = ImagePreprocessor(
    raw_annotation_path=ANNOTATIONS_PATH / "demo_annotations_sharks.csv",
    transform=transform_default,
    file_type="jpg",
    crop_type=None,
    model=model,
    device=device,
)

# Image factory to handle image creation and transformation
img_builder = ImageFactory(preprocessor)

# Unique object IDs in dataset
ids_all = preprocessor.img_annotations["object_id"].unique()

# Initialize timers for benchmarking
n = 10
total_img_time = 0.0
total_torch_time = 0.0
total_onnx_time = 0.0

# %% -------------------------
# Run inference benchmark
# -------------------------
for i in tqdm(range(n)):
    label = rd.randint(0, 1)  # random label (same or different object)

    # Randomly select example object ID
    example = rd.choice(ids_all)

    # Randomly select a different object ID
    error = rd.choice(ids_all[ids_all != example])

    # Fetch image rows for both objects
    rows_example = preprocessor.get_rows_img([example], "object_id")
    rd.shuffle(rows_example)  # shuffle images

    rows_error = preprocessor.get_rows_img([error], "object_id")
    rd.shuffle(rows_error)

    # -------------------------
    # Image loading and transformation
    # -------------------------
    img_start = time.time()
    if label:
        # Positive pair (same object)
        img_obj = img_builder.create_img(row_img=rows_example[0])
        img_obj.load_img()
        img_obj.transform_img()

        couple_obj = img_builder.create_img(row_img=rows_example[1])
        couple_obj.load_img()
        couple_obj.transform_img()
    else:
        # Negative pair (different objects)
        img_obj = img_builder.create_img(row_img=rows_example[0])
        img_obj.load_img()
        img_obj.transform_img()

        couple_obj = img_builder.create_img(row_img=rows_error[1])
        couple_obj.load_img()
        couple_obj.transform_img()
    img_end = time.time()
    total_img_time += img_end - img_start

    # Prepare input tensors
    img1 = img_obj.img_transform.to("cpu")
    img2 = couple_obj.img_transform.to("cpu")
    example_inputs = (img1.unsqueeze(0), img2.unsqueeze(0))

    # -------------------------
    # PyTorch inference
    # -------------------------
    with torch.no_grad():
        torch_start = time.time()
        torch_outputs = [
            (model(example_inputs[0], example_inputs[1])).numpy()
        ]  # run model
        torch_end = time.time()
        total_torch_time += torch_end - torch_start

    # -------------------------
    # ONNX inference
    # -------------------------
    onnx_start = time.time()
    onnxruntime_outputs = onnx_utils.run_onnx_inference(
        ort_session=ort_session, inputs=example_inputs
    )
    onnx_end = time.time()
    total_onnx_time += onnx_end - onnx_start

    # -------------------------
    # Verify outputs match
    # -------------------------
    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(
            torch.tensor(torch_output), torch.tensor(onnxruntime_output)
        )

# %% -------------------------
# Compute average times
# -------------------------
avg_img_time = total_img_time / n
avg_torch_time = total_torch_time / n
avg_onnx_time = total_onnx_time / n

print("\n--- Averages over {:,} iterations ---".format(n))
print(f"Image load and transform average time:  {avg_img_time:.4f}s")
print(f"PyTorch inference average time:         {avg_torch_time:.4f}s")
print(f"ONNX inference average time:            {avg_onnx_time:.4f}s")
# %%
