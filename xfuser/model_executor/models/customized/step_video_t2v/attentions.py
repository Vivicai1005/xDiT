import torch
import torch.nn as nn
from einops import rearrange

import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            print(f"{class_name}.{func.__name__} took {elapsed_time:.3f} seconds")
        else:
            print(f"{func.__name__} took {elapsed_time:.3f} seconds")
        return result
    return wrapper

try:
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except ImportError:
    xFuserLongContextAttention = None


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def attn_processor(self, attn_type):
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            return self.parallel_attn_func
        else:
            raise Exception('Not supported attention type...')

    @timing_decorator
    def torch_attn_func(
            self,
            q,
            k,
            v,
            attn_mask=None,
            causal=False,
            drop_rate=0.0,
            **kwargs
    ):

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)

        if attn_mask is not None and attn_mask.ndim == 3:
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        x = rearrange(x, 'b h s d -> b s h d')
        return x

    @timing_decorator
    def parallel_attn_func(
            self,
            q,
            k,
            v,
            causal=False,
            **kwargs
    ):
        assert xFuserLongContextAttention is not None;
        'to use sequence parallel attention, xFuserLongContextAttention should be imported...'
        hybrid_seq_parallel_attn = xFuserLongContextAttention()
        x = hybrid_seq_parallel_attn(
            None, q, k, v, causal=causal
        )
        return x
