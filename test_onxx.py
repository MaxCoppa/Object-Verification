# %%
import torch
import os
import time
import pandas as pd
from utils import image_utils, onnx_utils
from veri_models import ModelEnsembler, ObjectVeriSiamese
from veri_models.object_verif_model_pred import ObjectVeriSiameseMC
from veri_models.model_ensembler_pred import ModelEnsemblerMC
import torch
import random as rd

from image_general import ImagePreprocessor, ImageFactory

import torchvision.transforms.v2 as v2

# %%
start_load_model = time.time()

# model = ObjectVeriSiameseMC(backbone="efficientnet_v2_s")
# model.load_state_dict(torch.load("pretrained_model/comparaison/comparaison_67.pth"))

# model = ObjectVeriSiameseMC(backbone="resnet50")
# model.load_state_dict(torch.load("pretrained_model/comparaison/comparaison_test_last_model.pth"))

model1 = ObjectVeriSiameseMC(backbone="efficientnet_v2_s")
model1.load_state_dict(torch.load("pretrained_model/comparaison/comparaison_67.pth"))


model2 = ObjectVeriSiameseMC()
model2.load_state_dict(torch.load("pretrained_model/comparaison/comparaison_101.pth"))

model3 = ObjectVeriSiameseMC()
model3.load_state_dict(torch.load("pretrained_model/comparaison/comparaison_100.pth"))


model = ModelEnsemblerMC(
    models=[
        model1,
        model2,
        model3,
    ]
)
model.to("cpu")
model.eval()
end_load_model = time.time()

print(f"Model loading time: {end_load_model - start_load_model:.2f}s")

# %%
example_inputs = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
onnx_model_path = "pretrained_model/object_verif_mc_l.onnx"
# is_export = onnx_utils.export_model_onnx(
#     model=model, example_inputs=example_inputs, onnx_model_path=onnx_model_path
# )
# %%
start_onnx_load = time.time()
onnx_model = onnx_utils.load_model_onxx(onnx_model_path=onnx_model_path)
ort_session = onnx_utils.create_ort_session(onnx_model_path)
end_onnx_load = time.time()
print(
    f"ONNX model load + session creation time: {end_onnx_load - start_onnx_load:.2f}s"
)

# %%

file_type = "png"
crop_type = "object"
device = "cpu"

transform_default = image_utils.transform_fc("test")

preprocessor = ImagePreprocessor(
    raw_annotation_path="data/annotations/annotations_cleaned_pred_demo.csv",
    transform=transform_default,
    file_type=file_type,
    crop_type=crop_type,
    model=model,
    device=device,
)

img_builder = ImageFactory(preprocessor)
plates_all = preprocessor.img_annotations["plate"].unique()
n = 100
total_img_time = 0.0
total_torch_time = 0.0
total_onnx_time = 0.0

for i in range(n):
    if i % 100 == 0:
        print(f"{i}/{n}")
    label = rd.randint(0, 1)
    example = rd.choice(plates_all)
    error = rd.choice(plates_all)
    rows_example = preprocessor.get_rows_img([example], "plate")
    rows_error = preprocessor.get_rows_img([error], "plate")

    img_start = time.time()
    if label:
        img_obj = img_builder.create_img(row_img=rows_example[0])
        img_obj.load_img()
        img_obj.transform_img()

        couple_obj = img_builder.create_img(row_img=rows_example[1])
        couple_obj.load_img()
        couple_obj.transform_img()
    else:
        img_obj = img_builder.create_img(row_img=rows_example[0])
        img_obj.load_img()
        img_obj.transform_img()

        couple_obj = img_builder.create_img(row_img=rows_error[1])
        couple_obj.load_img()
        couple_obj.transform_img()

    img_end = time.time()
    total_img_time += img_end - img_start

    img1 = img_obj.img_transform.to("cpu")
    img2 = couple_obj.img_transform.to("cpu")
    example_inputs = (img1.unsqueeze(0), img2.unsqueeze(0))

    with torch.no_grad():
        torch_start = time.time()

        torch_outputs = [(model(example_inputs[0], example_inputs[1])).numpy()]
        torch_end = time.time()
        total_torch_time += torch_end - torch_start

    onnx_start = time.time()
    onnxruntime_outputs = onnx_utils.run_onnx_inference(
        ort_session=ort_session, inputs=example_inputs
    )
    onnx_end = time.time()
    total_onnx_time += onnx_end - onnx_start

    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(
            torch.tensor(torch_output), torch.tensor(onnxruntime_output)
        )


# %%
avg_img_time = total_img_time / n
avg_torch_time = total_torch_time / n
avg_onnx_time = total_onnx_time / n

print("\n--- Averages over {:,} iterations ---".format(n))
print(f"Image load and transform average time:  {avg_img_time:.4f}s")
print(f"PyTorch inference average time:         {avg_torch_time:.4f}s")
print(f"ONNX inference average time:            {avg_onnx_time:.4f}s")
# %%
