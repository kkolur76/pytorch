import torch
from torch import _prims
from torch._subclasses.fake_tensor import disable_fake_tensor_mode_tracing
from typing import Tuple
from torch.types import _device, _dtype
import torch._decomp as decomp


def _philox_rand(
    shape: torch.Size,
    seed: torch.Tensor,
    offset: torch.Tensor,
    stride: Tuple[int, ...],
    device: _device,
    dtype: _dtype,
):
    # FIXME - Need to add a nondeterministic_seeded tag to this op. Not sure how to do that yet.
    stride = tuple(stride)
    with torch.random.fork_rng(
        devices=[
            device,
        ]
    ):
        torch.manual_seed(seed)
        full_size = list(shape) + [stride[-1]]
        full_stride = stride + (1,)
        for i in reversed(range(len(full_stride))):
            if i == 0:
                full_size[i] = shape[0]
            else:
                assert full_stride[i - 1] % full_stride[i] == 0
                full_size[i] = full_stride[i - 1] // full_stride[i]

        for i in range(len(full_stride)):
            if offset % full_stride[i] == 0:
                full_size[i] += offset // full_stride[i]
                break
        else:
            assert False

        return torch.rand(full_size, device=device, dtype=dtype).as_strided(
            shape, stride, offset
        )


def _philox_rand_meta(
    shape: torch.Size,
    seed: torch.Tensor,
    offset: torch.Tensor,
    stride: Tuple[int, ...],
    device: _device,
    dtype: _dtype,
):
    # TODO - Update the state here
    # TODO - Directly running torch.rand is both inefficient and consumes
    # memory. For now, we are running torch.rand to find the total offset
    # consumed by this op. In future, we can replace this with the actual
    # formula used by the generators to calculate offset.

    # if device.type == "cuda":
    #     torch.manual_seed(0)
    #     torch.rand(shape, device, dtype)

    return _prims.TensorMeta(shape=shape, strides=stride, dtype=dtype, device=device)


def register_rng_prims():
    _prims._make_prim(
        schema="philox_rand(int[] size, Tensor seed, Tensor offset, int[] stride, Device? device=None, ScalarType? dtype=None) -> Tensor",
        return_type=_prims.RETURN_TYPE.NEW,
        meta=_philox_rand_meta,
        impl_aten=_philox_rand,
        tags=(torch.Tag.nondeterministic_seeded,),
        doc="",
    )


    
