# PyTorch/XLA SPMD المواضيع المتقدمة

في هذه الوثيقة، سنغطي بعض الموضوعات المتقدمة حول GSPMD. يرجى قراءة [دليل المستخدم SPMD](https://github.com/pytorch/xla/blob/master/docs/spmd_basic.md) قبل الانتقال إلى هذه الوثيقة.

### تحميل البيانات من المضيف إلى الجهاز مع مراعاة التجزئة

يأخذ PyTorch/XLA SPMD برنامجًا أحادي الجهاز، ويقسمه إلى أجزاء وينفذه بشكل متوازٍ. يتطلب التنفيذ SPMD استخدام PyTorch DataLoader الأصلي، والذي ينقل البيانات بشكل متزامن من المضيف إلى أجهزة XLA. وهذا يمنع التدريب أثناء نقل بيانات الإدخال في كل خطوة. لتحسين أداء تحميل البيانات الأصلي، جعلنا PyTorch/XLA ParallelLoader يدعم التجزئة المدخلة مباشرة (src)، عند تمرير وسيط kwarg الاختياري _input_sharding_:

```python
# MpDeviceLoader returns ParallelLoader.per_device_loader as iterator
train_loader = pl.MpDeviceLoader(
         train_loader,  # wraps PyTorch DataLoader
         device,
	     # assume 4d input and we want to shard at the batch dimension.
         input_sharding=xs.ShardingSpec(input_mesh, ('data', None, None, None)))
```

### تحسين الجهاز الظاهري

ينقل PyTorch/XLA عادة بيانات المصفوفة بشكل غير متزامر من المضيف إلى الجهاز بمجرد تحديد المصفوفة. وذلك لتداخل نقل البيانات مع وقت تتبع الرسم البياني. ومع ذلك، نظرًا لأن GSPMD يسمح للمستخدم بتعديل تجزئة المصفوفة بعد _تحديد المصفوفة، نحتاج إلى تحسين لمنع النقل غير الضروري لبيانات المصفوفة ذهابًا وإيابًا بين المضيف والجهاز. نقدم تحسين الجهاز الظاهري، وهي تقنية لوضع بيانات المصفوفة على جهاز ظاهري SPMD: 0 أولاً، قبل تحميلها إلى الأجهزة المادية عندما يتم الانتهاء من جميع قرارات التجزئة. يتم وضع كل بيانات المصفوفة في وضع SPMD على جهاز ظاهري، SPMD: 0. يتم عرض الجهاز الظاهري على المستخدم كجهاز XLA XLA: 0 مع الشرائح الفعلية على الأجهزة المادية، مثل TPU: 0، TPU: 1، وما إلى ذلك.

## شبكة هجينة

تُجرِّد الشبكة بشكل جميل كيفية بناء شبكة الأجهزة المادية. يمكن للمستخدمين ترتيب الأجهزة بأي شكل وترتيب باستخدام الشبكة المنطقية. ومع ذلك، يمكنك تحديد شبكة أكثر كفاءة بناءً على الطوبولوجيا المادية، خاصة عند مشاركة اتصالات شريحة Data Center Network (DCN) عبرها. تُنشئ HybridMesh شبكة توفر أداءً جيدًا خارج الصندوق لمثل هذه البيئات متعددة الشرائح. فهو يقبل ici_mesh_shape وdcn_mesh_shape والتي تشير إلى أشكال شبكة منطقية للأجهزة المتصلة داخليًا وخارجيًا.

```python
from torch_xla.distributed.spmd import HybridMesh

# This example is assuming 2 slices of v4-8.
# - ici_mesh_shape: shape of the logical mesh for inner connected devices.
# - dcn_mesh_shape: shape of logical mesh for outer connected devices.
ici_mesh_shape = (1, 4, 1) # (data, fsdp, tensor)
dcn_mesh_shape = (2, 1, 1)

mesh = HybridMesh(ici_mesh_shape, dcn_mesh_shape, ('data','fsdp','tensor'))
print(mesh.shape())
>> OrderedDict([('data', 2), ('fsdp', 4), ('tensor', 1)])
```

### تشغيل SPMD على TPU Pod

لا يلزم إجراء أي تغيير في التعليمات البرمجية للانتقال من مضيف TPU واحد إلى TPU Pod إذا قمت ببناء شبكة المواصفات والتقسيم الخاصة بك بناءً على عدد الأجهزة بدلاً من بعض الثوابت المرمزة ثابتة. لتشغيل حمل عمل PyTorch/XLA على TPU Pod، يرجى الرجوع إلى قسم [Pods](https://github.com/pytorch/xla/blob/master/docs/pjrt.md#pods) في دليل PJRT.

### XLAShardedTensor

`xs.mark_sharding` هي عملية في الموقع سترفق ملاحظة التجزئة بالمصفوفة المدخلة، ولكنها أيضًا تعيد كائن Python `XLAShardedTensor`.

يتمثل الاستخدام الأساسي لـ `XLAShardedTensor` [[RFC](https://github.com/pytorch/xla/issues/3871)] في إضافة تعليق إلى `torch.tensor` الأصلي (على جهاز واحد) باستخدام مواصفات التجزئة. يحدث التعليق على الفور، ولكن التجزئة الفعلية للمصفوفة يتم تأخيرها حيث يتم تنفيذ الحساب بشكل كسول، باستثناء مصفوفات الإدخال التي تتم تجزئتها دون تأخير. بمجرد إضافة تعليق على المصفوفة وتغليفها داخل `XLAShardedTensor`، يمكن تمريرها إلى عمليات PyTorch و`nn.Module` الطبقات كما `torch.Tensor`. هذا أمر مهم لضمان إمكانية تكديس نفس طبقات PyTorch وعمليات المصفوفة مع `XLAShardedTensor`. وهذا يعني أن المستخدم لا يحتاج إلى إعادة كتابة العمليات ورمز النموذج الموجودة لحساب التجزئة. وبشكل أكثر تحديدًا، فإن `XLAShardedTensor` سيحقق المتطلبات التالية:

* `XLAShardedTensor` هو فئة فرعية من `torch.Tensor` وتعمل مباشرة مع عمليات الشعلة الأصلية و`module.layers`. نحن نستخدم `__torch_dispatch__` لإرسال `XLAShardedTensor` إلى backend XLA. يقوم PyTorch/XLA باسترداد ملاحظات التجزئة المرفقة لتعقب الرسم البياني واستدعاء XLA SPMDPartitioner.
* داخليًا، يتم دعم `XLAShardedTensor` (ومدخلها global_tensor) بواسطة `XLATensor` بهيكل بيانات خاص يحتفظ بالإشارات إلى بيانات الشرائح على الجهاز.
* يمكن جمع المصفوفة المجزأة بعد التنفيذ الكسول وإعادتها إلى المضيف كمصفوفة global_tensor عند الطلب على المضيف (على سبيل المثال، طباعة قيمة المصفوفة العالمية).
* يتم تحويل المقابض إلى الشرائح المحلية بشكل صارم بعد التنفيذ الكسول. تعرض `XLAShardedTensor` [local_shards](https://github.com/pytorch/xla/blob/4e8e5511555073ce8b6d1a436bf808c9333dcac6/torch_xla/distributed/spmd/xla_sharded_tensor.py#L117) لإرجاع الشرائح المحلية على الأجهزة القابلة للعنونة كـ <code>List[[XLAShard](https://github.com/pytorch/xla/blob/4e8e5511555073ce8b6d1a436bf808c9333dcac6/torch_xla/distributed/spmd/xla_sharded_tensor.py#L12)]</code>.
هناك أيضًا جهد مستمر لدمج <code>XLAShardedTensor</code> في <code>DistributedTensor</code> API لدعم backend XLA [[RFC](https://github.com/pytorch/pytorch/issues/92909)].

### تكامل DTensor

أصدرت PyTorch نسخة تجريبية من [DTensor](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md) في 2.1.

نحن نقوم بتكامل PyTorch/XLA SPMD في DTensor API [RFC](https://github.com/pytorch/pytorch/issues/92909). لدينا تكامل مفهومي للدليل لـ `distribute_tensor`، والذي يستدعي واجهة برمجة التطبيقات للتعليق `mark_sharding` لتجزئة المصفوفة وحسابها باستخدام XLA:

```python
import torch
from torch.distributed import DeviceMesh, Shard, distribute_tensor

# distribute_tensor now works with `xla` backend using PyTorch/XLA SPMD.
mesh = DeviceMesh("xla", list(range(world_size)))
big_tensor = torch.randn(100000, 88)
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(0)])
```

هذه الميزة تجريبية، لذا يرجى الانتظار للحصول على تحديثات ومزيد من الأمثلة والبرامج التعليمية في الإصدارات القادمة.

### تجزئة التنشيط لـ torch.compile

في الإصدار 2.3، أضافت PyTorch/XLA عملية مخصصة `dynamo_mark_sharding` والتي يمكن استخدامها لأداء تجزئة التنشيط في منطقة `torch.compile`. هذا جزء من جهودنا المستمرة لجعل `torch.compile` + `GSPMD` الطريقة الموصى بها لأداء استدلال النموذج باستخدام PyTorch/XLA. مثال على استخدام هذه العملية المخصصة:

```
# Activation output sharding
device_ids = [i for i in range(self.num_devices)] # List[int]
mesh_shape = [self.num_devices//2, 1, 2] # List[int]
axis_names = "('data', 'model')" # string version of axis_names
partition_spec = "('data', 'model')" # string version of partition spec
torch.ops.xla.dynamo_mark_sharding(output, device_ids, mesh_shape, axis_names, partition_spec)
```

### أداة تصحيح SPMD

نوفر أداة `تصور موضع الشريحة` لمستخدم PyTorch/XLA SPMD على TPU/GPU/CPU مع single-host/multi-host: يمكنك استخدام `visualize_tensor_sharding` لتصور المصفوفة المجزأة، أو يمكنك استخدام `visualize_sharding` لتصور سلسلة التجزئة. فيما يلي مثالان للرمز على TPU single-host(v4-8) مع `visualize_tensor_sharding` أو `visualize_sharding`:

- مقتطف الشفرة المستخدمة `visualize_tensor_sharding` ونتيجة التصور:

```python
import rich

# Here, mesh is a 2x2 mesh with axes 'x' and 'y'
t = torch.randn(8, 4, device='xla')
xs.mark_sharding(t, mesh, ('x', 'y'))

# A tensor's sharding can be visualized using the `visualize_tensor_sharding` method
from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
generated_table = visualize_tensor_sharding(t, use_color=False)
```

<picture>
<source media="(prefers-color-scheme: dark)" srcset="_static/img/spmd_debug_1.png">
<img alt="visualize_tensor_sharding example on TPU v4-8(single-host)" src="_static/img/spmd_debug_1_light.png">
</picture>

- مقتطف الشفرة المستخدمة `visualize_sharding` ونتيجة التصور:

```python
from torch_xla.distributed.spmd.debugging import visualize_sharding
sharding = '{devices=[2,2]0,1,2,3}'
generated_table = visualize_sharding(sharding, use_color=False)
```

<picture>
<source media="(prefers-color-scheme: dark)" srcset="_static/img/spmd_debug_2.png">
<img alt="visualize_sharding example on TPU v4-8(single-host)" src="_static/img/spmd_debug_2_light.png">
</picture>

يمكنك استخدام هذه الأمثلة على TPU/GPU/CPU single-host وتعديلها لتشغيلها على multi-host. ويمكنك تعديلها لأسلوب التجزئة `tiled`، و`partial_replication`، و`replicated`.

### التجزئة التلقائية

نقدم ميزة جديدة لـ PyTorch/XLA SPMD، تسمى "التجزئة التلقائية"، [RFC](https://github.com/pytorch/xla/issues/6322). هذه ميزة تجريبية في `r2.3` و`nightly`، والتي تدعم `XLA:TPU` ومضيف TPUVM واحد.

يمكن تمكين التجزئة التلقائية لـ PyTorch/XLA بإحدى الطرق التالية:

- قم بتعيين متغير البيئة `XLA_AUTO_SPMD=1`
- استدعاء واجهة برمجة تطبيقات SPMD في بداية التعليمات البرمجية الخاصة بك:

```python
import torch_xla.runtime as xr
xr.use_spmd(auto=True)
```

- استدعاء `pytorch.distributed._tensor.distribute_module` مع `auto-policy` و`xla`:

```python
import torch_xla.runtime as xr
from torch.distributed._tensor import DeviceMesh, distribute_module
from torch_xla.distributed.spmd import auto_policy

device_count = xr.global_runtime_device_count()
device_mesh = DeviceMesh("xla", list(range(device_count)))

# Currently, model should be loaded to xla device via distribute_module.
model = MyModule()  # nn.module
sharded_model = distribute_module(model, device_mesh, auto_policy)
```

اختياريًا، يمكنك تعيين الخيارات/متغيرات البيئة التالية للتحكم في سلوك
تمرير التجزئة التلقائي القائم على XLA:

- `XLA_AUTO_USE_GROUP_SHARDING`: تجزئة إعادة تجميع المعلمات. تم الإعداد بشكل افتراضي.
- `XLA_AUTO_SPMD_MESH`: شكل شبكة منطقية لاستخدامها للتجزئة التلقائية. على سبيل المثال،
`XLA_AUTO_SPMD_MESH=2,2` يقابل شبكة 2x2 مع 4 أجهزة عالمية. إذا لم يتم تعيينه،
سيتم استخدام شكل شبكة جهاز افتراضي `num_devices,1`.