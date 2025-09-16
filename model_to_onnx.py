"""
Executable script to compare Object Verification Siamese Network with PyTorch and ONNX.
Supports command-line arguments with short flags.
"""

import argparse
import time
import random as rd
import torch
from tqdm import tqdm
from utils import image_utils, onnx_utils
from object_verif_models import ObjectVeriSiameseMC
from image_general import ImagePreprocessor, ImageFactory
from config import ANNOTATIONS_PATH, MODELS_PATH


def parse_args():
    """Parse command-line arguments with short flags."""
    parser = argparse.ArgumentParser(
        description="Create ONNX model and Compare model with PyTorch and ONNX."
    )

    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="model_1.pth",
        help="Path to pretrained PyTorch model",
    )
    parser.add_argument(
        "-o",
        "--onnx_path",
        type=str,
        default="model_1.onnx",
        help="Path to save/load ONNX model",
    )
    parser.add_argument(
        "-b",
        "--comparaison_torch_onnx",
        type=bool,
        default=False,
        help="Run the full comparaison loop (image loading + PyTorch + ONNX inference)",
    )
    return parser.parse_args()


def main(
    model_path="model_1.pth",
    onnx_path="model_1.onnx",
    b=False,
):
    """
    Create ONNX model and Compare model with PyTorch and ONNX for the Object Verification Siamese model.

    Args:
        iterations (int): Number of random image pairs to test
        file_type (str): Image file type
        model_path (str): Path to PyTorch model
        onnx_path (str): Path to save/load ONNX model
        b (bool): Whether to run the full comparaison loop
    """

    device = "cpu"  # Device here not cuda
    file_type = "jpg"  # Image file type
    iterations = 25  # Number of random image pairs to test ONNX VS Pytorch

    # -------------------------
    # Load PyTorch model
    # -------------------------
    start_load_model = time.time()
    model = ObjectVeriSiameseMC()
    model.load_state_dict(torch.load(MODELS_PATH / model_path))
    model.to(device)
    model.eval()
    end_load_model = time.time()
    print(f"PyTorch model loaded in {end_load_model - start_load_model:.2f}s")

    # -------------------------
    # Export to ONNX if needed
    # -------------------------
    example_inputs = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
    onnx_model_path = MODELS_PATH / onnx_path
    if not onnx_model_path.exists():
        onnx_utils.export_model_onnx(model, example_inputs, onnx_model_path)
        print(f"ONNX model exported to {onnx_model_path}")

    # Load ONNX model and create inference session
    start_onnx_load = time.time()
    onnx_model = onnx_utils.load_model_onnx(onnx_model_path)
    ort_session = onnx_utils.create_ort_session(onnx_model_path)
    end_onnx_load = time.time()
    print(f"ONNX load + session creation time: {end_onnx_load - start_onnx_load:.2f}s")

    if not b:
        print(
            "Torch comparaison ONNX skipped. Use --comparaison_torch_onnx to run the full loop."
        )
        return

    # -------------------------
    # Setup image preprocessor
    # -------------------------
    transform_default = image_utils.transform_fc("test")
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

    # Initialize timers
    total_img_time = 0.0
    total_torch_time = 0.0
    total_onnx_time = 0.0

    # -------------------------
    # Comparaison loop
    # -------------------------
    for _ in tqdm(range(iterations)):
        label = rd.randint(0, 1)
        example = rd.choice(ids_all)
        error = rd.choice(ids_all[ids_all != example])

        rows_example = preprocessor.get_rows_img([example], "object_id")
        rd.shuffle(rows_example)
        rows_error = preprocessor.get_rows_img([error], "object_id")
        rd.shuffle(rows_error)

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

        img1 = img_obj.img_transform.to(device)
        img2 = couple_obj.img_transform.to(device)
        example_inputs = (img1.unsqueeze(0), img2.unsqueeze(0))

        # PyTorch inference
        with torch.no_grad():
            torch_start = time.time()
            torch_outputs = [(model(example_inputs[0], example_inputs[1])).numpy()]
            torch_end = time.time()
            total_torch_time += torch_end - torch_start

        # ONNX inference
        onnx_start = time.time()
        onnx_outputs = onnx_utils.run_onnx_inference(ort_session, example_inputs)
        onnx_end = time.time()
        total_onnx_time += onnx_end - onnx_start

        # Verify outputs match
        assert len(torch_outputs) == len(onnx_outputs)
        for t_out, o_out in zip(torch_outputs, onnx_outputs):
            torch.testing.assert_close(torch.tensor(t_out), torch.tensor(o_out))

    # -------------------------
    # Print Comparaison results
    # -------------------------
    print("\n--- Averages over {:,} iterations ---".format(iterations))
    print(f"Image load & transform: {total_img_time / iterations:.4f}s")
    print(f"PyTorch inference:      {total_torch_time / iterations:.4f}s")
    print(f"ONNX inference:         {total_onnx_time / iterations:.4f}s")


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        onnx_path=args.onnx_path,
        b=args.comparaison_torch_onnx,
    )
