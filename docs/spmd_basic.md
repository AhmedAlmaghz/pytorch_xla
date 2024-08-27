# PyTorch/XLA SPMD الدليل الإرشادي للمستخدم

في هذا الدليل، نناقش كيفية دمج GSPMD في PyTorch/XLA، ونقدم نظرة عامة على التصميم لتوضيح كيفية عمل واجهة برمجة التطبيقات الخاصة بتشذيب SPMD وبنيتها.

## ما هو PyTorch/XLA SPMD؟

GSPMD هو نظام موازاة تلقائي لأحمال العمل الشائعة في ML. سيقوم مترجم XLA بتحويل برنامج الجهاز الفردي إلى برنامج مقسم مع مجموعات مناسبة، بناءً على تلميحات التشذيب المقدمة من المستخدم. تتيح هذه الميزة للمطورين كتابة برامج PyTorch كما لو كانت على جهاز واحد كبير دون أي عمليات حسابية مخصصة للتشذيب و/أو اتصالات مجمعة للتوسع.

![alt_text](_static/img/spmd_mode.png "image_tooltip")

_الشكل 1. مقارنة بين استراتيجيتي تنفيذ مختلفتين، (أ) لغير SPMD و(ب) لـ SPMD._

## كيفية استخدام PyTorch/XLA SPMD؟

فيما يلي مثال بسيط على استخدام SPMD

```python
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

# تمكين وضع التنفيذ XLA SPMD.
xr.use_spmd()

# شبكة الأجهزة، وهذا ومخطط التقسيم وكذلك شكل tensor المدخلة يحدد شكل الشريحة الفردية.
num_devices = xr.global_runtime_device_count()
mesh_shape = (num_devices, 1)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('data', 'model'))

t = torch.randn(8, 4).to(xm.xla_device())

# تقسيم الشبكة، يحتفظ كل جهاز بـ 1/8 من الإدخال
partition_spec = ('data', 'model')
xs.mark_sharding(t, mesh, partition_spec)
```

دعنا نشرح هذه المفاهيم واحدة تلو الأخرى

### وضع SPMD

لاستخدام SPMD، يجب تمكينه عبر `xr.use_spmd()`. في وضع SPMD، هناك جهاز منطقي واحد فقط. تتم معالجة الحساب الموزع والمجموعات بواسطة `mark_sharding`. لاحظ أنه لا يمكن للمستخدم خلط SPMD مع مكتبات موزعة أخرى.

### الشبكة

بالنسبة لمجموعة معينة من الأجهزة، تكون الشبكة المادية عبارة عن تمثيل لطوبولوجيا الاتصال.

1. `mesh_shape` عبارة عن مجموعة يتم ضربها في العدد الإجمالي للأجهزة الفعلية.
2. `device_ids` هو دائمًا تقريبًا `np.array(range(num_devices))`.
3. يُنصح المستخدمون أيضًا بإعطاء اسم لكل بُعد من أبعاد الشبكة. في المثال أعلاه، البعد الأول للشبكة هو بُعد `data` والبعد الثاني للشبكة هو بُعد `model`.

يمكنك أيضًا التحقق من مزيد من معلومات الشبكة عبر

```
>>> mesh.shape()
OrderedDict([('data', 4), ('model', 1)])
```

### مواصفات التقسيم

partition_spec لها نفس الرتبة مثل tensor المدخلة. يصف كل بُعد كيفية تشذيب البعد المقابل لtensor المدخلة عبر شبكة الأجهزة. في المثال أعلاه، يتم تشذيب البعد الأول لـ tensor `t` في بُعد `data` ويتم تشذيب البعد الثاني في بُعد `model`.

يمكن للمستخدم أيضًا تشذيب tensor الذي له أبعاد مختلفة من شكل الشبكة.

```python
t1 = torch.randn(8, 8, 16).to(device)
t2 = torch.randn(8).to(device)

# يتم تكرار البعد الأول.
xs.mark_sharding(t1, mesh, (None, 'data', 'model'))

# يتم تشذيب البعد الأول في بُعد البيانات.
# يتم استخدام بُعد النموذج للتكرار عند حذفه.
xs.mark_sharding(t2, mesh, ('data',))

# يتم تشذيب البعد الأول عبر كلا محوري الشبكة.
xs.mark_sharding( t2, mesh, (('data', 'model'),))
```

## قراءة إضافية

1. [مثال](https://github.com/pytorch/xla/blob/master/examples/data_parallel/train_resnet_spmd_data_parallel.py) لاستخدام SPMD للتعبير عن التوازي في البيانات.
2. [مثال](https://github.com/pytorch/xla/blob/master/examples/fsdp/train_decoder_only_fsdp_v2.py) لاستخدام SPMD للتعبير عن FSDP (Fully Sharded Data Parallel).
3. [موضوعات متقدمة في SPMD](https://github.com/pytorch/xla/blob/master/docs/spmd_advanced.md)
4. [نقطة تفتيش موزعة SPMD](https://github.com/pytorch/xla/blob/master/docs/spmd_distributed_checkpoint.md)