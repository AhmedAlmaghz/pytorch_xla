# Fully Sharded Data Parallel(FSDP) عبر SPMD

Fully Sharded Data Parallel عبر SPMD أو FSDPv2 هي أداة لإعادة صياغة خوارزمية FSDP الشهيرة في SPMD. [هذا](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/spmd_fully_sharded_data_parallel.py) هو ميزة تجريبية تهدف إلى تقديم واجهة مألوفة للمستخدمين للاستفادة من جميع المزايا التي توفرها SPMD. ويمكن الاطلاع على وثيقة التصميم [هنا](https://github.com/pytorch/xla/issues/6379).

يرجى مراجعة [دليل مستخدم SPMD](./spmd_basic.md) قبل المتابعة. يمكنك أيضًا العثور على مثال قابل للتشغيل الأدنى [هنا](https://github.com/pytorch/xla/blob/master/examples/fsdp/train_decoder_only_fsdp_v2.py).

مثال على الاستخدام:

```python3
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2

# تحديد شبكة الأجهزة باتباع الممارسة الشائعة في SPMD
num_devices = xr.global_runtime_device_count()
mesh_shape = (num_devices, 1)
device_ids = np.array(range(num_devices))
# تجدر الإشارة إلى أن شبكة الأجهزة يجب أن تحتوي على محور باسم 'fsdp'، والذي سيتم تقسيم الأوزان والتنشيطات عليه.
mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))

# تقسيم المدخلات، وافتراض أن x هو مصفوفة ثنائية الأبعاد.
x = xs.mark_sharding(x, mesh, ('fsdp', None))

# كما هو الحال في FSDP العادية، ولكن هناك حاجة إلى شبكة أجهزة إضافية.
model = FSDPv2(my_module, mesh)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
output = model(x, y)
loss = output.sum()
loss.backward()
optim.step()
```

من الممكن أيضًا تقسيم الطبقات الفردية بشكل منفصل وإضافة غلاف خارجي للتعامل مع أي معلمات متبقية. فيما يلي مثال على التغليف التلقائي لكل طبقة `DecoderLayer`.

```python3
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Apply FSDP sharding on each DecoderLayer layer.
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        decoder_only_model.DecoderLayer
    },
)
model = FSDPv2(
    model, mesh=mesh, auto_wrap_policy=auto_wrap_policy)
```

## تقسيم الإخراج

لضمان تنفيذ مترجم XLA لخوارزمية FSDP بشكل صحيح، نحتاج إلى تقسيم كل من الأوزان والتنشيطات. وهذا يعني تقسيم إخراج طريقة التقديم. نظرًا لأن إخراج دالة التقديم يمكن أن يختلف، فإننا نقدم shard_output لتقسيم التنشيطات في الحالات التي لا تندرج فيها وحدة الإخراج الخاصة بك ضمن إحدى هذه الفئات:

1. مصفوفة واحدة
2. مجموعة من المصفوفات حيث العنصر 0 هو التنشيط

مثال على الاستخدام:

```python3
def shard_output(output, mesh):
    xs.mark_sharding(output.logits, mesh, ('fsdp', None, None))

model = FSDPv2(my_module, mesh, shard_output)
```

## التحقق من نقطة التدرج

حاليًا، يجب تطبيق التحقق من نقطة التدرج على الوحدة النمطية قبل غلاف FSDP. وإلا، فإن الحلقة المتكررة بشكل متكرر في الوحدات النمطية الفرعية ستنتهي في حلقة لا نهائية. سنقوم بإصلاح هذه المشكلة في الإصدارات المستقبلية.

مثال على الاستخدام:

```python3
from torch_xla.distributed.fsdp import checkpoint_module

model = FSDPv2(checkpoint_module(my_module), mesh)
```

## مثال HuggingFace Llama 2

لدينا نسخة من HF Llama 2 لتوضيح التكامل المحتمل [هنا](https://github.com/huggingface/transformers/compare/main...pytorch-tpu:transformers:llama2-spmd-fsdp).