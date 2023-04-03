# Owner(s): ["module: functorch"]

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    xfail_inherited_tests,
)

from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes
from functorch.compile import aot_function, aot_module, draw_graph, print_compile, nop


class TestFunctionalizationRngOps(TestCase):
    @dtypes(torch.float32)
    def test_forward(self, dtype, device):
        def fn(x):
            a = torch.rand_like(x) * x
            return a

        x = torch.rand(10, device=device, dtype=dtype)
        
        for seed in range(10):
            torch.cuda.manual_seed(seed)
            ref = fn(x)

            torch.cuda.manual_seed(seed)
            aot_fn = aot_function(fn, nop)
            res = aot_fn(x)

            self.assertEqual(ref, res)



    @dtypes(torch.float32)
    def test_autograd_function(self, dtype, device):
        shape = (16, 16)

        class Custom(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                a = torch.rand_like(x) * x
                return a

            @staticmethod
            def backward(ctx, grad_out):
                x, = ctx.saved_tensors
                return grad_out * torch.rand_like(grad_out) * torch.cos(x)

        custom = Custom.apply

        x = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)

        x_clone = x.clone().detach().requires_grad_(True)

        torch.cuda.manual_seed(123)
        ref = custom(x)
        ref.sum().backward()

        torch.cuda.manual_seed(123)
        aot_custom = aot_function(custom, nop)
        res = aot_custom(x_clone)
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)


only_for = ("cuda",)
instantiate_device_type_tests(TestFunctionalizationRngOps, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
