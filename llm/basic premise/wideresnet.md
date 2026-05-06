---
data: 2026-05-06
---
# 1.

1. `nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]`定义通道数 是个数组
2. 通道的合法性检查
```python
assert((depth - 4) % 6 == 0)  
n = (depth - 4) / 6
```
3. `_class_ torch.nn.Conv2d(_in_channels_, _out_channels_, _kernel_size_, _stride=1_, _padding=0_, _dilation=1_, _groups=1_, _bias=True_, _padding_mode='zeros'_, _device=None_, _dtype=None_)
   `self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,padding=1, bias=False)`
   3\*32\*32->16*
   
4. 