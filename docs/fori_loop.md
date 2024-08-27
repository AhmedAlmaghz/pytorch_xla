# `While_loop` لتحسين استخدام الذاكرة وتجمیعها

### `while_loop`
يستبدل `while_loop` حلقة `while` البايثونية النقية، ويدعم PyTorch حلقة `while_loop` من خلال [torch._higher_order_ops.while_loop](https://github.com/pytorch/pytorch/blob/62311257adb902d6a4ea98809c88895af1dbbf2b/torch/_higher_order_ops/while_loop.py#L66).
يوفر PyTorch/XLA دعمًا تجريبيًا لواجهة XLA الخلفية لـ `torch._higher_order_ops.while_loop` عبر `XLA::While`.

#### الاستخدام:

```python
import torch_xla.experimental.fori_loop
from torch._higher_order_ops.while_loop import while_loop
result = while_loop(cond_fn, body_fn, init)
```
- `cond_fn`: دالة شرط معرفة من قبل المستخدم.
- `body_fn`: دالة جسم الحلقة معرفة من قبل المستخدم.
- `init`: القيم الأولية (مصفوفة أو قائمة).

#### مثال بسيط باستخدام `while_loop`:

```py
# PJRT_DEVICE=TPU python
>>> import torch
>>> import torch_xla
>>> import torch_xla.experimental.fori_loop
>>> from torch._higher_order_ops.while_loop import while_loop
>>> import torch_xla.core.xla_model as xm
>>>
>>> device = xm.xla_device()
>>>
>>> def cond_fn(iteri, x):
...   return iteri > 0
...
>>> def body_fn(iteri, x):
...   return iteri - 1, torch.add(x, 1)
...
>>> init_val = torch.tensor(3, device=device)
>>> iteri = torch.tensor(10, device=device)
>>> _, res = while_loop(cond_fn, body_fn, (iteri, init_val))
>>> res
FunctionalTensor(lvl=0, value=
tensor(13, device='xla:0'))
```

## حالة اختبار مجموعة التحكم
لمقارنة أفضل للفرق بين `pure python while loop` و`while_loop`، هناك حالة اختبار واحدة تسمى حلقة `while` بايثون النقية بمنطق مشابه: التراكم بالإضافة إلى 1 لعشر مرات:

### مثال مجموعة التحكم باستخدام حلقة `while` بايثون النقية

```py
# PJRT_DEVICE=TPU python
>>> import torch
>>> import torch_xla
>>> import torch_xla.core.xla_model as xm
>>>
>>> device = xm.xla_device()
>>>
>>> init_val = torch.tensor(1, device=device)
>>> iteri = torch.tensor(50, device=device)
>>>
>>> while iteri > 0:
...   init_val = init_val + 1
...   iteri -= 1
...
>>> init_val
tensor(51, device='xla:0')
```
سيقوم PyTorch/XLA بتضمين دعم `while_loop` في الإصدار 2.4 مع حالة الاختبار، وسيتم إضافة الدعم لـ `fori_loop` بعد الإصدار 2.4. بالنسبة لـ `while_loop`، يجب أن نقوم حاليًا بتحديد `body_fn` بنفس شكل `input` و`output(return args)`