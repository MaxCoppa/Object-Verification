import torch
import os
import pandas as pd
import torch
import onnx
import onnxruntime


def export_model_onnx(model, example_inputs, onnx_model_path):
    """
    Export the PyTorch model to ONNX format
    """

    torch.onnx.export(
        model,
        example_inputs,
        onnx_model_path,
        export_params=True,  # store the trained parameters
        do_constant_folding=True,  # optimize constants
        input_names=["input"],  # name of input tensor
        output_names=["output"],  # name of output tensor
    )
    return True


def load_model_onxx(onnx_model_path):
    """
    Load and check the ONNX model
    """

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    return onnx_model


def create_ort_session(onnx_model_path):
    """
    Create an ONNX Runtime InferenceSession for running inference.
    """
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
    )

    return ort_session


def run_onnx_inference(ort_session, inputs):
    """
    Run inference on an ONNX model using ONNX Runtime.

    """

    onnx_inputs = [tensor.numpy(force=True) for tensor in inputs]

    onnxruntime_input = {
        input_arg.name: input_value
        for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)
    }

    # ONNX Runtime returns a list of outputs
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    return onnxruntime_outputs
