import torch
import torch._decomp as decomp
from torch._subclasses.fake_tensor import disable_fake_tensor_mode_tracing
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing


aten = torch.ops.aten
rng_decompositions = {}


def register_decomposition(aten_op):
    return decomp.register_decomposition(aten_op, rng_decompositions)


def get_default_stride(size):
    """
    A helper function to get the strides for a contiguous tensor of a given
    shape.
    """
    stride = [1] * len(size) + [1]
    for idx in reversed(range(len(size))):
        stride[idx] = stride[idx + 1] * size[idx]
    stride = stride[1:]
    return stride


@register_decomposition(aten.rand)
def rand(shape, dtype=None, layout=torch.strided, device=None, pin_memory=False):
    device = device or "cpu"
    seed, offset = PhiloxStateTracker.get_state_as_tuple()

    numel = 1
    for dim_size in shape:
        numel *= dim_size
    PhiloxStateTracker.advance_offset(numel)
    dtype = dtype or torch.float32
    stride = get_default_stride(shape)
    philox_rand = torch.ops.prims.philox_rand
    r = philox_rand(shape, seed, offset, stride, device, dtype)
    return r

    # return out_grad * (1 - y * y).conj_physical()


@register_decomposition(aten.rand_like)
def rand_like(
    x: torch.Tensor,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=torch.preserve_format,
):
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    PhiloxStateTracker.advance_offset(x.numel())
    philox_rand = torch.ops.prims.philox_rand
    r = philox_rand(x.shape, seed, offset, x.stride(), x.device, x.dtype)
    return r


class PhiloxState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.seed = None
        self.base_offset = None
        self.relative_offset = 0

    def set_relative_offset(self, new_offset):
        self.relative_offset = new_offset

    def advance_offset(self, consumed_offset):
        self.relative_offset += consumed_offset

    def set_state(self, seed, base_offset, relative_offset):
        self.seed = seed
        self.base_offset = base_offset
        self.relative_offset = relative_offset

    def get_state_as_tuple(self):
        with disable_fake_tensor_mode_tracing():
            return (self.seed, self.base_offset + self.relative_offset)

    def get_state_as_tensor(self):
        with disable_fake_tensor_mode_tracing():
            seed_portion = self.seed.reshape(1)
            offset_portion = (self.base_offset + self.relative_offset).reshape(1)
            return torch.cat([seed_portion, offset_portion])

    def set_state_from_tensor(self, state):
        with disable_fake_tensor_mode_tracing():
            seed, offset = torch.split(state, 1)
            self.seed = seed[0]
            self.base = offset[0]
            self.relative_offset = 0


class PhiloxStateTracker:
    running_state = None
    fwd_state = None
    bwd_state = None

    @classmethod
    def reset(cls):
        cls.running_state = PhiloxState()
        cls.fwd_state = PhiloxState()
        cls.bwd_state = PhiloxState()

    @classmethod
    def mark_beginning_of_forward(cls):
        cls.running_state = cls.fwd_state

    @classmethod
    def mark_beginning_of_backward(cls):
        cls.running_state = cls.bwd_state

    @classmethod
    def record_state(cls, seed, offset, mode):
        if mode == "forward":
            cls.fwd_state.set_state(seed, offset, 0)
        else:
            assert mode == "backward"
            cls.bwd_state.set_state(seed, offset, 0)

    @classmethod
    def get_state_as_tensor(cls):
        return cls.running_state.get_state_as_tensor()

    @classmethod
    def get_state_as_tuple(cls):
        return cls.running_state.get_state_as_tuple()

    @classmethod
    def advance_torch_state_after_fwd(cls):
        print(f"forward total offset = {cls.fwd_state.relative_offset}")
        cls.advance_torch_state(cls.fwd_state.relative_offset)

    @classmethod
    def advance_torch_state_after_bwd(cls):
        print(f"backward total offset = {cls.bwd_state.relative_offset}")
        cls.advance_torch_state(cls.bwd_state.relative_offset)

    @classmethod
    def advance_torch_state(cls, offset):
        rng_state = torch.cuda.get_rng_state()
        seed = rng_state[800:808].view(dtype=torch.int64)[0]
        offset = rng_state[808:].view(dtype=torch.int64)[0]
        new_offset = offset + offset
        torch.cuda.set_rng_state(cls.create_rng_state_tensor(seed, new_offset))

    @classmethod
    def set_state_from_tensor(cls, x):
        cls.running_state.set_state_from_tensor(x)

    @staticmethod
    def create_rng_state_tensor(seed, offset):
        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        prefix = torch.tensor([-1] * 800, dtype=torch.uint8)
        return torch.cat([prefix, seed_portion, offset_portion])

    @classmethod
    def advance_offset(cls, consumed_offset):
        cls.running_state.advance_offset(consumed_offset)

    @staticmethod
    def get_offset_jump(shape):
        # TODO - Specific to PyTorch CUDA impl. It calculates the total number
        # of randoms generated by CUDA. If everything fits nicely in the
        # stride-loop CUDA kernel, this is equal to the number of elements. But,
        # when a thread block has some unusable threads, it can be a different
        # number.

        # For impl, look at calc_execution_policy
        numel = 1
        for dim_size in shape:
            numel *= dim_size

        block_size = 256
        unroll = 4
        curand4_engine_calls = 4
        device_property = torch.cuda.get_device_properties(torch.cuda.current_device())
        blocks_per_sm = int(
            device_property.max_threads_per_multi_processor / block_size
        )
        grid_size = int((numel + block_size - 1) / block_size)
        grid_size = min(
            grid_size, device_property.multi_processor_count * blocks_per_sm
        )
        offset = (
            int((numel - 1) / (block_size * grid_size * unroll) + 1)
            * curand4_engine_calls
        )
        print(numel, offset)
        return offset


class RNGStateHelper:
    @staticmethod
    def get_torch_state_as_tuple():
        with disable_proxy_modes_tracing():
            with disable_fake_tensor_mode_tracing():
                rng_state = torch.cuda.get_rng_state()
                seed = rng_state[800:808].view(dtype=torch.int64)[0]
                offset = rng_state[808:].view(dtype=torch.int64)[0]
                return seed, offset