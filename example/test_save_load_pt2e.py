# Summary:
# - We can save and reload the int8 weights.
# - The `use_reference_representation=True` will decompose the quantization primitives into more low-level operations.


import copy
import itertools
from enum import Enum

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torch.nn as nn
from torch.ao.quantization import ObserverBase
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    QUANT_ANNOTATION_KEY,
    X86InductorQuantizer,
)
from torch.export import export_for_training
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skipIfNoInductorSupport,
    skipIfNoX86,
)
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import skipIfTorchDynamo


def get_file_size(file_path):
    import os

    size_in_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    return size_in_bytes / 1024 / 1024


def get_model_size(model):
    model_size = 0
    for key, val in model.state_dict().items():
        model_size += val.numel() * val.element_size()
    return model_size / 1024 / 1024


import pytest


@pytest.mark.parametrize(
    "use_reference_representation",
    [True, False],
    ids=["use_reference_representation=True", "use_reference_representation=False"],
)
def test_save_load(use_reference_representation):
    class SingleLinearModule(torch.nn.Module):
        def __init__(self, use_bias=False) -> None:
            super().__init__()
            self.linear = nn.Linear(1024, 1024 * 1024, bias=use_bias)

        def forward(self, x):
            return self.linear(x)

    m = SingleLinearModule()
    example_inputs = (torch.randn(1, 1024),)
    quantizer = X86InductorQuantizer().set_global(
        xiq.get_default_x86_inductor_quantization_config()
    )
    # program capture

    m = export_for_training(
        m,
        example_inputs,
    ).module()

    # QAT Model failed to deepcopy
    is_qat = False
    export_model = m if is_qat else copy.deepcopy(m)
    m = export_model.eval()
    m = prepare_qat_pt2e(m, quantizer) if is_qat else prepare_pt2e(m, quantizer)
    # Calibrate
    m(*example_inputs)
    prepare_model = copy.deepcopy(m)
    m = convert_pt2e(m, use_reference_representation=use_reference_representation)
    print(f"Converted model: ")
    m.print_readable()
    convert_model = copy.deepcopy(m)
    print(f"Convert model size: {get_model_size(convert_model)} MB")
    quantized_ep = torch.export.export(convert_model, example_inputs)
    fname = "ep_converted_model.pt2"
    # rm `linear.weight`
    if "linear.weight" in quantized_ep.state_dict:
        quantized_ep.state_dict.pop("linear.weight")
    # quantized_ep.state_dict.pop("linear.weight")
    torch.export.save(quantized_ep, fname)
    ep_size = get_file_size(fname)
    print(f"Saved model size: {ep_size} MB")
    loaded_ep = torch.export.load(fname)
    loaded_quantized_model = loaded_ep.module()
    from torch._inductor import config as inductor_config

    inductor_config.freezing = True
    loaded_quantized_model = torch.compile(loaded_quantized_model)
    res = loaded_quantized_model(*example_inputs)
    res = loaded_quantized_model(*example_inputs)


"""

pytest -sv ./test_save_load_pt2e.py 
====================================================================================== test session starts =======================================================================================
platform linux -- Python 3.11.10, pytest-8.3.3, pluggy-1.5.0 -- /home/yliu7/miniforge3/envs/test/bin/python3.11
cachedir: .pytest_cache
rootdir: /home/yliu7/workspace/inc/3rd-party/pytorch
configfile: pytest.ini
plugins: cov-5.0.0, xdist-3.6.1, typeguard-4.3.0
collecting ... Fail to import hypothesis in common_utils, tests are not derandomized
collected 2 items                                                                                                                                                                                

test_save_load_pt2e.py::test_save_load[use_reference_representation=True] Converted model: 
class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[1, 1024]"; 
    
        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        # No stacktrace found for following nodes
        _scale_0 = self._scale_0
        _zero_point_0 = self._zero_point_0
        _frozen_param0 = self._frozen_param0
        quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor(x, 0.026026815176010132, 127, 0, 255, torch.uint8);  x = None
        
         # File: /home/yliu7/workspace/inc/3rd-party/pytorch/example/test_save_load_pt2e.py:55 in forward, code: return self.linear(x)
        dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor(quantize_per_tensor_default, 0.026026815176010132, 127, 0, 255, torch.uint8);  quantize_per_tensor_default = None
        
         # File: /home/yliu7/miniforge3/envs/test/lib/python3.11/site-packages/torch/ao/quantization/pt2e/representation/rewrite.py:707 in _reference_dequantize_per_channel_int8, code: x_i8 = torch.ops.aten.clamp(x_i8, quant_min, quant_max)
        clamp: "i8[1, 3, 3, 3]" = torch.ops.aten.clamp.default(_frozen_param0, -128, 127);  _frozen_param0 = None
        
         # File: /home/yliu7/miniforge3/envs/test/lib/python3.11/site-packages/torch/ao/quantization/pt2e/representation/rewrite.py:708 in _reference_dequantize_per_channel_int8, code: x_i8 = torch.transpose(x_i8, ch_axis, -1)
        transpose: "i8[1, 3, 3, 3]" = torch.ops.aten.transpose.int(clamp, 0, -1);  clamp = None
        
         # File: /home/yliu7/miniforge3/envs/test/lib/python3.11/site-packages/torch/ao/quantization/pt2e/representation/rewrite.py:709 in _reference_dequantize_per_channel_int8, code: x_i32 = x_i8.to(torch.int32)
        to: "i32[1, 3, 3, 3]" = torch.ops.aten.to.dtype(transpose, torch.int32);  transpose = None
        
         # File: /home/yliu7/miniforge3/envs/test/lib/python3.11/site-packages/torch/ao/quantization/pt2e/representation/rewrite.py:710 in _reference_dequantize_per_channel_int8, code: out_fp32 = (x_i32 - zero_points).to(torch.float) * scales
        sub: "i32[1, 3, 3, 3]" = torch.ops.aten.sub.Tensor(to, _zero_point_0);  to = _zero_point_0 = None
        to_1: "f32[1, 3, 3, 3]" = torch.ops.aten.to.dtype(sub, torch.float32);  sub = None
        mul: "f32[1, 3, 3, 3]" = torch.ops.aten.mul.Tensor(to_1, _scale_0);  to_1 = _scale_0 = None
        
         # File: /home/yliu7/miniforge3/envs/test/lib/python3.11/site-packages/torch/ao/quantization/pt2e/representation/rewrite.py:711 in _reference_dequantize_per_channel_int8, code: out_fp32 = torch.transpose(out_fp32, ch_axis, -1)
        transpose_1: "f32[1, 3, 3, 3]" = torch.ops.aten.transpose.int(mul, 0, -1);  mul = None
        
         # File: /home/yliu7/workspace/inc/3rd-party/pytorch/example/test_save_load_pt2e.py:55 in forward, code: return self.linear(x)
        linear: "f32[1, 1048576]" = torch.ops.aten.linear.default(dequantize_per_tensor_default, transpose_1);  dequantize_per_tensor_default = transpose_1 = None
        return pytree.tree_unflatten((linear,), self._out_spec)
        
Convert model size: 1036.0 MB
Saved model size: 1036.020471572876 MB
PASSED
test_save_load_pt2e.py::test_save_load[use_reference_representation=False] Converted model: 
class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[1, 1024]"; 
    
        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        # No stacktrace found for following nodes
        _scale_0 = self._scale_0
        _zero_point_0 = self._zero_point_0
        _frozen_param0 = self._frozen_param0
        
         # File: /home/yliu7/workspace/inc/3rd-party/pytorch/example/test_save_load_pt2e.py:55 in forward, code: return self.linear(x)
        dequantize_per_channel_default = torch.ops.quantized_decomposed.dequantize_per_channel.default(_frozen_param0, _scale_0, _zero_point_0, 0, -128, 127, torch.int8);  _frozen_param0 = _scale_0 = _zero_point_0 = None
        
        # No stacktrace found for following nodes
        quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.022096268832683563, 134, 0, 255, torch.uint8);  x = None
        
         # File: /home/yliu7/workspace/inc/3rd-party/pytorch/example/test_save_load_pt2e.py:55 in forward, code: return self.linear(x)
        dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default, 0.022096268832683563, 134, 0, 255, torch.uint8);  quantize_per_tensor_default = None
        linear: "f32[1, 1048576]" = torch.ops.aten.linear.default(dequantize_per_tensor_default, dequantize_per_channel_default);  dequantize_per_tensor_default = dequantize_per_channel_default = None
        return pytree.tree_unflatten((linear,), self._out_spec)
        
Convert model size: 5132.0 MB
Saved model size: 1036.014588356018 MB

"""
