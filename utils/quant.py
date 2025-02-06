import torch
def quantize_tg(x, scale=0.1, zero_point=10, quantizer=None):
    if quantizer is not None:
        # print("run torch quantizer")
        if not x.is_quantized:
            x = quantizer(x)
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, scale=0.1, zero_point=10, dtype=torch.quint8)
        return x
    if not x.is_quantized:
        x = torch.quantize_per_tensor(x, scale=0.1, zero_point=10, dtype=torch.quint8)
    return x
def dequantize_tg(x, dequantizer=None):
    if dequantizer is not None:
        # print("run torch dequantizer")
        if x.is_quantized:
            x = dequantizer(x)
        if x.is_quantized:
            x = x.dequantize()
        return x
    if x.is_quantized:
        x = x.dequantize()
    assert not x.is_quantized
    return x
def quant_add(x, y, scale=0.1, zero_point=10, quantizer=None):
    if not x.is_quantized:
        x = quantize_tg(x, quantizer=quantizer)
        assert x.is_quantized
    if not y.is_quantized:
        y = quantize_tg(y, quantizer=quantizer)
        assert y.is_quantized
    
    return torch.ops.quantized.add(x, y, scale, zero_point)
def force_dequantize(x):
    if x.is_quantized:
        x = x.dequantize()
    assert not x.is_quantized
    return x
def quant_sub(x, y, scale=0.1, zero_point=10):
    if not x.is_quantized:
        x = quantize_tg(x)
    if y.is_quantized:
        y = dequantize_tg(y)
        y = -y
        y = quantize_tg(y)
    else:
        y = -y
        y = quantize_tg(y)
    return torch.ops.quantized.add(x, y, scale, zero_point)
def quant_mul(x, y, quantizer = None, dequantizer = None):
    # y is scalar
    # if y is a tensor, change to scalar
    if type(y) == torch.Tensor and y.numel() == 1:
        y = y.item()
        if not x.is_quantized:
            x = quantize_tg(x, quantizer=quantizer)
        return torch.ops.quantized.mul(x, y)
    elif type(y) == int or type(y) == float:
        if not x.is_quantized:
            x = quantize_tg(x, quantizer=quantizer)
        return torch.ops.quantized.mul(x, y)
    else:
        if x.is_quantized:
            x = dequantize_tg(x, dequantizer=dequantizer)
        if y.is_quantized:
            y = dequantize_tg(y, dequantizer=dequantizer)
        res = x*y
        return quantize_tg(res, quantizer=quantizer)

def quant_pow(x, y, quantizer = None, dequantizer = None):
    # y is scalar
    # if y is a tensor, change to scalar
    if type(y) == torch.Tensor:
        y = y.item()
    if x.is_quantized:
        x = dequantize_tg(x, dequantizer=dequantizer)
    res = x ** y
    return quantize_tg(res, quantizer=quantizer)

def recursive_apply(x, func):
    if isinstance(x, torch.Tensor):
        return func(x)
    elif isinstance(x, list):
        return [recursive_apply(item, func) for item in x]
    elif isinstance(x, tuple):
        return tuple(recursive_apply(item, func) for item in x)
    else:
        return x