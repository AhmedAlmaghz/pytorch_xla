# Custom GPU Kernels عبر Triton

يدعم PyTorch/XLA الآن [Triton](https://openai.com/research/triton) kernels، مما يمكّن من تنفيذ نموذج التعلم العميق عالي الأداء على وحدات معالجة الرسومات (GPUs). Triton، وهي لغة ومترجم متخصصان لبرمجة GPU، تمكن المطورين من كتابة نوى مخصصة تستفيد من الإمكانات الكاملة لوحدات معالجة الرسومات (GPUs) لمختلف العمليات في نماذج التعلم العميق.

نظرًا لوجود نواة Triton محددة على النحو التالي:
```python3
@triton.jit
def add_kernel(
    x_ptr، # *Pointer* إلى أول متجه إدخال.
    y_ptr، # *Pointer* إلى متجه الإدخال الثاني.
    output_ptr، # *Pointer* إلى متجه الإخراج.
    n_elements، # حجم المتجه.
    BLOCK_SIZE: tl.constexpr، # عدد العناصر التي يجب أن تعالجها كل برنامج.
    # ملاحظة: `constexpr` حتى يمكن استخدامه كقيمة شكل.
):
  # نواة Triton الإضافية من https://github.com/openai/triton/blob/main/python/tutorials/01-vector-add.py#L28
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0، BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets، mask=mask)
  y = tl.load(y_ptr + offsets، mask=mask)
  output = x + y
  tl.store(output_ptr + offsets، output، mask=mask)

```

يمكننا تشغيل هذه النواة كجزء من رسم PyTorch/XLA البياني على النحو التالي:

```python3
import torch

import torch_xla.experimental.triton as xla_triton
import torch_xla

import triton
import triton.language as tl

size = 16
x = torch.arange(size, dtype=torch.int64).to("xla")
y = torch.arange(size, dtype=torch.int64).to("xla")
output = torch.empty_like(x)
block_size = 8
grid = (triton.cdiv(size, block_size),)

# triton_call takes the same arguments as the triton.jit function, in addition 
to the kernel itself and the grid that is used to execute the kernel.
All the tl.constexpr terms are passed as kwargs at the end.
payload = xla_triton.triton_call(
    x, y, output, size, kernel=add_kernel, grid=grid, BLOCK_SIZE=block_size)

# To make the triton kernel, a part of the PyTorch/XLA graph, we create a
# custom call node with the expected inputs, payload from triton_call,
# the output shapes and output dtypes. The payload already contains information
# regarding how the GPU buffers will be loaded when this node is executed.
output = torch_xla._XLAC._xla_gpu_custom_call([x, y], payload,
                                                [output.shape], [torch.int64])

```

بالنسبة للنوى الأكثر تعقيدًا، يمكنك أيضًا الرجوع إلى اختبار نواة Triton Flash Attention في PyTorch/XLA.

## التبعيات
تعتمد تكامل Triton على حزمة `triton` للعمل. تم اختبار هذا الكود مع `triton==2.3.0`. لتثبيت:
```bash
pip install --no-deps triton==2.3.0
```