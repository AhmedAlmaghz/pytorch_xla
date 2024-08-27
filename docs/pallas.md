# النوى المخصصة عبر Pallas
مع ظهور OpenAI [triton](https://openai.com/research/triton)، أصبحت النوى المخصصة أكثر شعبية في مجتمع GPU، على سبيل المثال، تقديم [FlashAttention](https://github.com/Dao-AILab/flash-attention) و [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html). من أجل تحقيق التكافؤ في ميزة في عالم TPU، قدمت Google [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html). لكي تواصل PyTorch/XLA دفع الأداء في TPU، يجب أن ندعم النوى المخصصة، وأفضل طريقة للقيام بذلك هي من خلال Pallas. وثيقة التصميم [TBA]().

لنفترض أن لديك نواة Pallas معرفة كما يلي:

```python3
from torch_xla.experimental.custom_kernel import jax_import_guard
jax_import_guard()

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(add_vectors_kernel,
                        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
                        )(x, y)
```

من المهم جدًا ملاحظة أنه يجب تشغيل `jax_import_guard()` قبل استيراد أي وحدات نمطية Jax. وإلا، سيتوقف البرنامج على TPU لأن Jax سيقفل TPU ولن تتمكن torch-xla من الوصول إليه.

## تبن النواة أعلاه لتكون متوافقة مع PyTorch/XLA

مثال الاستخدام:

```python3
q = torch.randn(3, 2, 128, 4).to("xla")
k = torch.randn(3, 2, 128, 4).to("xla")
v = torch.randn(3, 2, 128, 4).to("xla")

# تبن أي نواة Pallas
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
pt_kernel = make_kernel_from_pallas(add_vectors, lambda x, y: [(x.shape, x.dtype)])
output = pt_kernel(q, k)
```

بالنسبة للنوى البسيطة، يكون التبني بسيطًا مثل سطر واحد. بالنسبة للنوى الأكثر تعقيدًا، يمكنك الرجوع إلى تنفيذ Flash Attention للحصول على التفاصيل.

## استخدام النوى المدمجة
بالإضافة إلى لف Pallas الخارجية يدويًا، هناك نوى مدمجة حيث تم تنفيذ عمليات التبني بالفعل بواسطة PyTorch/XLA. يمكن استخدام هذه النوى المدمجة مثل أي نواة أخرى من torch.ops. النوى المدمجة الحالية المدعومة هي:

- FlashAttention
- PagedAttention

### FlashAttention

#### مثال الاستخدام

```python3
# استخدم النوى المدمجة
import torch_xla.experimental.custom_kernel
output = flash_attention(q, k, v)
```

#### مثال التكامل

لدينا مثال على [تكامل FlashAttention هنا](https://github.com/pytorch/xla/blob/master/examples/flash_attention/train_decoder_only_flash_attention.py) في نص اختبار التدريب الخاص بنا.

### PagedAttention

#### مثال الاستخدام

```python3
# استخدم النوى المدمجة
import torch_xla.experimental.custom_kernel
output = torch.ops.xla.paged_attention(
  query.squeeze(dim=1),
  key_cache,
  value_cache,
  context_lens,
  block_tables,
  pages_per_compute_block,
  megacore_mode=None,
)
```

#### مثال التكامل

يستخدم تكامل TPU vLLM [PagedAttention هنا](https://github.com/vllm-project/vllm/blob/f5e1bf5d44877149eaabf9c04379a4e14a023145/vllm/attention/backends/pallas.py#L194) لإدارة الذاكرة الفعالة مع ذاكرة التخزين المؤقت KV.

## التبعيات

يعتمد تكامل Pallas على JAX للعمل. ومع ذلك، فإن إصدار JAX غير متوافق مع إصدار PyTorch/XLA المثبت. لتثبيت JAX المناسب:

```bash
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
```