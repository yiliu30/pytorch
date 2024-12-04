###############################################################################
# Summay: It cann't handle the data that didn't seen during calibration.'
# Maybe a better logging is needed to understand the issue.
###############################################################################


import copy
import os
import random
import sys

import numpy as np
import pytest
import torch
from habana_frameworks.torch.core.quantizer import (
    _mark_nodes_as_annotated,
    _update_input_qspec_map,
    habana_quant_config_symmetric,
    habana_quantizer,
)
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import fga_assert_helper, is_gaudi1
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
    get_input_act_qspec,
    get_weight_qspec,
)
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


# Fixture to set the environment variable
# @pytest.fixture
# def set_env_variable():
#     variable_name_fx_pass = "USE_FX_GRAPH_PATTERN_MATCHING"
#     os.environ[variable_name_fx_pass] = "1"
#     # Yield to provide the value for the test
#     yield "1"
#     os.environ[variable_name_fx_pass] = "0"

# set_env_variable()

variable_name_fx_pass = "USE_FX_GRAPH_PATTERN_MATCHING"
os.environ[variable_name_fx_pass] = "1"


class SimpleModel(torch.nn.Module):
    def __init__(self, dtype):
        super(SimpleModel, self).__init__()
        self.gemm1 = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        out = self.gemm1(x)
        out = self.relu1(out)
        return out


class SimpleModelWithMultipleGraphs(torch.nn.Module):
    def __init__(self, dtype):
        super(SimpleModelWithMultipleGraphs, self).__init__()
        self.gemm1 = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.relu1 = torch.nn.ReLU()
        self.gemm2 = torch.nn.Linear(2, 2, dtype=dtype)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.gemm1(x)
        out = self.relu1(out)
        torch._dynamo.graph_break()
        out = self.gemm2(out)
        out = self.relu2(out)
        return out


def get_sample_model(test_case, quant_dtype, graph_breaks=False):
    dtype = torch.float32 if quant_dtype == torch.int8 else torch.bfloat16
    if test_case == "linear_relu":
        return (
            SimpleModelWithMultipleGraphs(dtype) if graph_breaks else SimpleModel(dtype)
        )


def get_sample_input(test_case, quant_dtype):
    CPU = torch.device("cpu")
    dtype = torch.float32 if quant_dtype == torch.int8 else torch.bfloat16
    if test_case == "linear_relu":
        return torch.randn(2, 4, device=CPU, dtype=dtype)


test_case_list = [
    "linear_relu",
]
quant_int_dtype_list = [
    torch.int8,
]
quant_float_dtype_list = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


def verify_nodes(ops_summary, expected_op_count):
    for op, count_list in expected_op_count.items():
        if not op.startswith("skip_"):
            fga_assert_helper(ops_summary=ops_summary, op=op, count_list=count_list)


class ThreeWaysModel(torch.nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, dim * 2)
        self.lin3 = torch.nn.Linear(dim, dim * 3)

    def forward(self, x):
        if x.sum() > 0:
            return self.lin1(x)
        elif x.sum() < 0:
            return self.lin2(x)
        else:
            return self.lin3(x)


import habana_frameworks.torch.core as htcore

htcore.hpu_set_env()

# Stabilizing testing.
torch.manual_seed(0xDEADDEAD)
random.seed(0xDEADDEAD)
np.random.seed(0xDEADDEAD)
torch.use_deterministic_algorithms(True)

CPU = torch.device("cpu")

dim = 10
inputs0 = torch.randn(4, dim)
inputs1 = -inputs0
inputs2 = inputs1 * 0

model = ThreeWaysModel(dim)
model.eval()

HPU = torch.device("hpu")

inputs0 = inputs0.to(HPU)
inputs1 = inputs1.to(HPU)
inputs2 = inputs2.to(HPU)
# inputs2 = inputs2.to(HPU)


model.to(device=HPU)
model.eval()

example_inputs0 = [
    inputs0,
]
example_inputs1 = [
    inputs1,
]
example_inputs2 = [
    inputs2,
]


# compile_model = torch.compile(model, backend="hpu_backend")
# compile_model(*example_inputs0)
# compile_model(*example_inputs1)
# compile_model(*example_inputs2)
# breakpoint()
quantizer = habana_quantizer()
quant_config = habana_quant_config_symmetric(torch.float8_e5m2)
quantizer.set_global(quant_config)
import logging

# logger = logging.getLogger(__file__)
from loguru import logger

with torch.no_grad():
    from torch._export import capture_pre_autograd_graph

    # if pass_input_during_export:
    #     model = capture_pre_autograd_graph(model, example_inputs0)
    # else:
    model = capture_pre_autograd_graph(model)

    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e

        model = prepare_pt2e(model, quantizer)
        logger.warning(f"After prepare.......")
        # calibrate
        calibrate_result = model(*example_inputs0)
        logger.warning(f"After calibrate input0")
        calibrate_result = model(*example_inputs1)
        logger.warning(f"After calibrate input1")

    # if use_graph_break:
    #     verify_nodes(fga.get_ops_summary(), expected_op_count["after_prepare_pt2e"])

    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        from torch.ao.quantization.quantize_pt2e import convert_pt2e

        model = convert_pt2e(model)
        # run inference with quantized model
        logger.warning(f"After convert")
        hpu_result0 = model(*example_inputs0)
        logger.warning(f"After infer with input 0")
        hpu_result1 = model(*example_inputs1)
        logger.warning(f"After infer with input 1")
        hpu_result2 = model(*example_inputs2)
        logger.warning(f"After infer with input 1")
        print(hpu_result2)

    htcore.hpu_reset_env()
# def use_pt2e_quant_flow(
#     test_case, quant_dtype, quantizer, expected_op_count, use_graph_break, pass_input_during_export
# ):
#     import habana_frameworks.torch.core as htcore

#     htcore.hpu_set_env()

#     # Stabilizing testing.
#     torch.manual_seed(0xDEADDEAD)
#     random.seed(0xDEADDEAD)
#     np.random.seed(0xDEADDEAD)
#     torch.use_deterministic_algorithms(True)

#     CPU = torch.device("cpu")
#     inputs0 = get_sample_input(test_case, quant_dtype)
#     inputs1 = get_sample_input(test_case, quant_dtype)
#     inputs2 = get_sample_input(test_case, quant_dtype)
#     example_inputs0 = [
#         inputs0,
#     ]
#     example_inputs1 = [
#         inputs1,
#     ]
#     example_inputs2 = [
#         inputs2,
#     ]

#     model = get_sample_model(test_case, quant_dtype, use_graph_break)
#     model.eval()

#     cpu_result2 = model(*example_inputs2)
#     print(cpu_result2)

#     HPU = torch.device("hpu")
#     inputs0 = inputs0.to(HPU)
#     inputs1 = inputs1.to(HPU)
#     inputs2 = inputs2.to(HPU)
#     example_inputs0 = [
#         inputs0,
#     ]
#     example_inputs1 = [
#         inputs1,
#     ]
#     example_inputs2 = [
#         inputs2,
#     ]

#     model.to(device=HPU)
#     model.eval()

#     with torch.no_grad():
#         from torch._export import capture_pre_autograd_graph

#         if pass_input_during_export:
#             model = capture_pre_autograd_graph(model, example_inputs0)
#         else:
#             model = capture_pre_autograd_graph(model)

#         with FxGraphAnalyzer(reset_dynamo=False) as fga:
#             from torch.ao.quantization.quantize_pt2e import prepare_pt2e

#             model = prepare_pt2e(model, quantizer)
#             # calibrate
#             calibrate_result = model(*example_inputs0)
#             calibrate_result = model(*example_inputs1)

#         if use_graph_break:
#             verify_nodes(fga.get_ops_summary(), expected_op_count["after_prepare_pt2e"])

#         with FxGraphAnalyzer(reset_dynamo=False) as fga:
#             from torch.ao.quantization.quantize_pt2e import convert_pt2e

#             model = convert_pt2e(model)
#             # run inference with quantized model
#             hpu_result2 = model(*example_inputs2)
#             print(hpu_result2)

#         if use_graph_break:
#             verify_nodes(fga.get_ops_summary(), expected_op_count["after_convert_pt2e"])
#             assert torch.allclose(cpu_result2[0].float(), hpu_result2[0].to(CPU).float(), rtol=1e-2, atol=1e-2)
#         else:
#             assert torch.allclose(cpu_result2[0].float(), hpu_result2[0].to(CPU).float(), rtol=2e-2, atol=2e-2)

#     htcore.hpu_reset_env()


"""

<pt261> user4@vm:pytest_working$ PT_HPU_LAZY_MODE=0  p graph_break.py 
Calling add_step_closure function does not have any effect. It's lazy mode only functionality. (warning logged once)
Calling mark_step function does not have any effect. It's lazy mode only functionality. (warning logged once)
Calling iter_mark_step function does not have any effect. It's lazy mode only functionality. (warning logged once)
/usr/lib/python3.10/inspect.py:288: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  return isinstance(object, types.FunctionType)
WARNING: The experimental weight sharing feature is enabled and may cause larger device memory
              consumption in quantized models. Please disable it by setting PT_HPU_WEIGHT_SHARING=0
libibverbs: Warning: couldn't open config directory '/tmp/tmp.B0MJyq88xA/build/etc/libibverbs.d'.
Calling add_step_closure function does not have any effect. It's lazy mode only functionality. (warning logged once)
Calling mark_step function does not have any effect. It's lazy mode only functionality. (warning logged once)
Calling iter_mark_step function does not have any effect. It's lazy mode only functionality. (warning logged once)
============================= HABANA PT BRIDGE CONFIGURATION =========================== 
 PT_HPU_LAZY_MODE = 0
 PT_RECIPE_CACHE_PATH = 
 PT_CACHE_FOLDER_DELETE = 0
 PT_HPU_RECIPE_CACHE_CONFIG = 
 PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
 PT_HPU_LAZY_ACC_PAR_MODE = 1
 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
 PT_HPU_EAGER_PIPELINE_ENABLE = 1
 PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE = 1
---------------------------: System Configuration :---------------------------
Num CPU Cores : 24
CPU RAM       : 82353568 KB
------------------------------------------------------------------------------
2024-12-04 17:21:59.717 | WARNING  | __main__:<module>:200 - After prepare.......
/home/venvvv/lib/python3.10/site-packages/torch/jit/_check.py:178: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
  warnings.warn(
/home/venvvv/lib/python3.10/site-packages/torch/jit/_check.py:178: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
  warnings.warn(
/home/venvvv/lib/python3.10/site-packages/torch/jit/_check.py:178: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
  warnings.warn(
2024-12-04 17:22:00.418 | WARNING  | __main__:<module>:203 - After calibrate input0
/home/venvvv/lib/python3.10/site-packages/torch/jit/_check.py:178: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
  warnings.warn(
2024-12-04 17:22:00.592 | WARNING  | __main__:<module>:205 - After calibrate input1
2024-12-04 17:22:00.593 | WARNING  | __main__:<module>:215 - After convert
/home/venvvv/lib/python3.10/site-packages/torch/jit/_check.py:178: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
  warnings.warn(
/home/venvvv/lib/python3.10/site-packages/torch/jit/_check.py:178: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
  warnings.warn(
/home/venvvv/lib/python3.10/site-packages/torch/jit/_check.py:178: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
  warnings.warn(
2024-12-04 17:22:00.877 | WARNING  | __main__:<module>:217 - After infer with input 0
/home/venvvv/lib/python3.10/site-packages/torch/jit/_check.py:178: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
  warnings.warn(
2024-12-04 17:22:00.957 | WARNING  | __main__:<module>:219 - After infer with input 1
Traceback (most recent call last):
  File "/home/user4/workspace/inc-fork/3rd-party/pytorch-integration/tests/pytest_working/graph_break.py", line 220, in <module>
    hpu_result2 = model(*example_inputs2)
  File "/home/venvvv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/venvvv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/venvvv/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 465, in _fn
    return fn(*args, **kwargs)
  File "/home/venvvv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/venvvv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user4/workspace/inc-fork/3rd-party/pytorch-integration/tests/pytest_working/graph_break.py", line 126, in forward
    def forward(self, x):
  File "/home/user4/workspace/inc-fork/3rd-party/pytorch-integration/tests/pytest_working/graph_break.py", line 127, in torch_dynamo_resume_in_forward_at_127
    if x.sum() > 0:
  File "/home/user4/workspace/inc-fork/3rd-party/pytorch-integration/tests/pytest_working/graph_break.py", line 129, in torch_dynamo_resume_in_forward_at_129
    elif x.sum() < 0:
  File "/home/venvvv/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 632, in _fn
    return fn(*args, **kwargs)
  File "/home/venvvv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1100, in forward
    return compiled_fn(full_args)
  File "/home/venvvv/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 321, in runtime_wrapper
    all_outs = call_func_at_runtime_with_args(
  File "/home/venvvv/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/utils.py", line 124, in call_func_at_runtime_with_args
    out = normalize_as_list(f(args))
  File "/home/venvvv/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 667, in inner_fn
    outs = compiled_fn(args)
  File "/home/venvvv/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 488, in wrapper
    return compiled_fn(runtime_args)
  File "/home/venvvv/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/utils.py", line 98, in g
    return f(*args)
  File "/home/venvvv/lib/python3.10/site-packages/habana_frameworks/torch/core/quantize_pt2e.py", line 256, in __call__
    raise NotImplementedError("Attempt to convert an unprepared module!.")
NotImplementedError: Attempt to convert an unprepared module!.
Exception ignored in: <function FxGraphAnalyzer.__del__ at 0x7fa86b869c60>
Traceback (most recent call last):
  File "/home/venvvv/lib/python3.10/site-packages/habana_frameworks/torch/utils/debug/dynamo_utils.py", line 52, in __del__
AttributeError: 'NoneType' object has no attribute 'unregister'

"""
