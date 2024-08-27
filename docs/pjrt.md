## وقت تشغيل PJRT

تمت هجرة PyTorch/XLA من وقت تشغيل XRT القائم على TensorFlow إلى [وقت تشغيل PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt) الذي يستخدمه [JAX](https://github.com/google/jax).

إذا صادفتك مشكلة مع PJRT، يرجى تقديم تقرير عن المشكلة على GitHub مع علامة "وقت التشغيل".

## الميزات الجديدة في PyTorch/XLA r2.1:

- أصبح PJRT مستقرًا في PyTorch/XLA r2.1!
- انتقلت واجهات برمجة التطبيقات العامة للوقت التشغيل من `torch_xla.experimental.pjrt` إلى `torch_xla.runtime`.
- تمت إعادة تسمية طريقة المبادرة `pjrt://` إلى `xla://`، وهي مسجلة بواسطة `torch_xla.distributed.xla_backend`.
- لا تزال الأسماء السابقة `torch_xla.experimental.*` متوفرة في هذا الإصدار للتوافق.
- أصبح `torchrun` مدعومًا الآن عند استخدام `init_method='xla://'`.
- إضافات جديدة لXPU وNeuron عبر واجهة برمجة تطبيقات C في PJRT.

## الميزات الجديدة في PyTorch/XLA r2.0:

- سيتم تكوين PJRT بشكل افتراضي إذا لم تقم بإدخال أي تكوين وقت تشغيل آخر. إذا استمررت في تعيين تكوين XRT (`XRT_TPU_CONFIG`)، فلن يكون لهذا التغيير أي تأثير.
- تحسين الأداء بنسبة تصل إلى 30% باستخدام تنفيذ وقت تشغيل TPU الجديد في `libtpu`.
- تنفيذ جديد لـ `xm.rendezvous` يمكنه التوسع إلى آلاف من أنوية TPU.
- [تجريبي] دعم `torch.distributed` لـ TPU v2 وv3، بما في ذلك `pjrt://` `init_method`.

## خلاصة القول

- لاستخدام وقت تشغيل معاينة PJRT، قم بتعيين متغير البيئة `PJRT_DEVICE` إلى `CPU` أو `TPU` أو `CUDA`.
- في XRT، تكون جميع الأحمال العملة الموزعة متعددة العمليات، مع وجود عملية واحدة لكل جهاز. على TPU v2 وv3 في PJRT، تكون الأحمال العملة متعددة العمليات ومتعددة الخيوط (4 عمليات مع خيطين لكل منها)، لذلك يجب أن يكون حمل العمل الخاص بك آمنًا للخيوط. راجع [تعدد الخيوط على TPU v2/v3](#multithreading-on-tpu-v2v3) و[قسم تعدد العمليات في دليل واجهة برمجة التطبيقات](https://github.com/pytorch/xla/blob/master/API_GUIDE.md#running-on-multiple-xla-devices-with-multi-processing) لمزيد من المعلومات. الاختلافات الرئيسية التي يجب مراعاتها:
- لتهيئة نموذج بطريقة آمنة للخيوط، قم ببث المعلمات عبر النسخ المتماثلة بعد التهيئة (`torch_xla.experimental.pjrt.broadcast_master_param`) أو قم بتحميل معلمات النسخة المتماثلة لكل منها من نقطة تفتيش مشتركة.
- بالنسبة لتوليد الأرقام العشوائية الأخرى، استخدم `torch.Generator` حيثما أمكن ذلك. إن مولد RNG العالمي لـ `torch` ليس آمنًا للخيوط، حتى إذا قمت بتعيين نفس `torch.manual_seed` عبر النسخ المتماثلة.
- لاستخدام `torch.distributed`، قم باستيراد `torch_xla.experimental.pjrt_backend` واستخدم `xla://` `init_method`.
- تعد هذه الخطوات اختيارية لـ GPU وTPU v4.

مثال على الفرق من XRT إلى PJRT:

```diff
 import os

 import torch
 import torch.nn as nn
 from torch.nn.parallel import DistributedDataParallel as DDP
 import torch.optim as optim
 import torch.distributed as dist
 import torch_xla
 import torch_xla.core.xla_model as xm
 import torch_xla.distributed.parallel_loader as pl
 import torch_xla.distributed.xla_backend
+import torch_xla.runtime as xr


 def _mp_fn(index):
   device = xm.xla_device()
-  dist.init_process_group('xla', rank=xr.global_ordinal(), world_size=xr.world_size())
+  dist.init_process_group('xla', init_method='xla://')

   torch.manual_seed(42)
   model = nn.Linear(128, 10).to(device)

+  # Optional for TPU v4 and GPU
+  xm.broadcast_master_param(model)
   model = DDP(model, gradient_as_bucket_view=True)

   loss_fn = nn.MSELoss()
   optimizer = optim.SGD(model.parameters(), lr=.001)

   for i in range(10):
     data, target = torch.randn((128, 128), device=device), torch.randn((128, 10), device=device)

     optimizer.zero_grad()
     output = model(data)
     loss = loss_fn(output, target)
     loss.backward()

     optimizer.step()
     xm.mark_step()

   # Print mean parameters so we can confirm they're the same across replicas
   print([p.mean() for p in model.parameters()])

 if __name__ == '__main__':
-  os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
-  os.environ['MASTER_ADDR'] = 'localhost'
-  os.environ['MASTER_PORT'] = '12355'

+  # Recommended: set PJRT_DEVICE to your local device type
+  os.environ['PJRT_DEVICE'] = 'TPU'

   torch_xla.launch(_mp_fn)
```

## الفوائد

- تكوين وقت تشغيل بسيط: ما عليك سوى تعيين `PJRT_DEVICE` إلى `TPU` أو `CPU` أو `CUDA` وبدء استخدام XLA! أو، دع PJRT يختار جهازًا تلقائيًا بناءً على بيئتك.
- تحسين الأداء: تقليل النفقات العامة من gRPC يعني تنفيذ أسرع من النهاية إلى النهاية. في TorchBench 2.0، لاحظنا تحسنًا بنسبة تزيد عن 35% في وقت التدريب على TPU v4.
- سهولة تنفيذ pod: ما عليك سوى نسخ رمزك إلى كل عامل TPU، وتشغيلها جميعًا في نفس الوقت باستخدام `gcloud compute tpus tpuvm ssh --worker=all`.
- تحسين التوسع: يزيل [قيود XRT على أحجام المعلمات](https://github.com/pytorch/xla/pull/3920) ويدعم ما يصل إلى 2048 شريحة TPU.

## البدء السريع

للبدء في استخدام PJRT مع PyTorch/XLA، كل ما عليك فعله هو تعيين متغير البيئة `PJRT_DEVICE`. إذا كنت تعمل على TPU v2 أو v3، فاستمر في القراءة لمعرفة الاختلافات بين TPU v2 وv3 وv4.

### وحدة المعالجة المركزية

على أي جهاز مثبت عليه PyTorch/XLA، يمكنك تشغيل مثال MNIST الخاص بنا على وحدة المعالجة المركزية كما يلي:

```
PJRT_DEVICE=CPU python3 xla/test/test_train_mp_mnist.py --fake_data
```

### TPU

لإنشاء TPU جديد مثبت عليه PyTorch/XLA r2.0:

```
gcloud alpha compute tpus tpu-vm create $USER-pjrt --accelerator-type=v4-8 --version=tpu-vm-v4-pt-2.0 --zone=us-central2-b --project=$PROJECT
```

على v4-8، يمكنك تشغيل مثال ResNet50 الخاص بنا كما يلي:

```
git clone --depth=1 --branch r2.0 https://github.com/pytorch/xla.git
PJRT_DEVICE=TPU python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=256 --num_epochs=1
```

بشكل افتراضي، سيستخدم PJRT جميع شرائح TPU. لاستخدام شريحة TPU واحدة فقط، قم بتكوين `TPU_PROCESS_BOUNDS` و`TPU_VISIBLE_CHIPS`:

```
TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=256 --num_epochs=1
```

#### مجموعات

على مجموعات TPU، استخدم `gcloud` لتشغيل أمرك على كل TPU بشكل متوازٍ:

```
gcloud alpha compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command="git clone --depth=1 --branch r1.13 https://github.com/pytorch/xla.git"
gcloud alpha compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command="PJRT_DEVICE=TPU python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=256 --num_epochs=1"
```

#### Docker

يمكنك أيضًا استخدام Docker لتشغيل حمل العمل الخاص بك في حاوية مع تثبيت مسبق لـ PyTorch/XLA:

```
export DOCKER_IMAGE=gcr.io/...

# Optional: authenticate docker if your image is in a private GCP repository
gcloud compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command "sudo gcloud auth configure-docker"

# Run your workload
gcloud compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command "sudo docker run --rm --privileged --net=host -e PJRT_DEVICE=TPU $DOCKER_IMAGE python pytorch/xla/test/test_train_mp_imagenet.py --fake_data"
```

لاحظ أن `docker run` يتطلب الوصول المتميز إلى المضيف (`--privileged`) لتعريض جهاز TPU للحاوية. يتم دعم Docker على مجموعات TPU فقط مع شبكة المضيف `--net=host` في هذا الوقت. راجع [وثائق Cloud TPU](https://cloud.google.com/tpu/docs/run-in-container) لمزيد من المعلومات.

### GPU

### التدريب على عقدة GPU واحدة

لاستخدام وحدات معالجة الرسومات مع PJRT، ما عليك سوى تعيين `PJRT_DEVICE=CUDA` وتهيئة `GPU_NUM_DEVICES` إلى عدد الأجهزة الموجودة على المضيف. على سبيل المثال:

```
PJRT_DEVICE=CUDA GPU_NUM_DEVICES=4 python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=128 --num_epochs=1
```

يمكنك أيضًا استخدام `torchrun` لبدء التدريب متعدد وحدات معالجة الرسومات (GPU) على عقدة واحدة. على سبيل المثال،

```
PJRT_DEVICE=CUDA torchrun --nnodes 1 --nproc-per-node ${NUM_GPU_DEVICES} xla/test/test_train_mp_imagenet.py --fake_data --pjrt_distributed --batch_size=128 --num_epochs=1
```

في المثال أعلاه، يعني `--nnodes` عدد الأجهزة (الأجهزة المادية أو الافتراضية) التي سيتم استخدامها (فهو 1 نظرًا لأننا نقوم بالتدريب على عقدة واحدة). يشير `--nproc-per-node` إلى عدد أجهزة GPU التي سيتم استخدامها.

### التدريب متعدد العقد على GPU

**ملاحظة: تعمل هذه الميزة فقط لـ CUDA 12 أو أحدث**. على غرار كيفية استخدام PyTorch للتدريب متعدد العقد، يمكنك تشغيل الأمر كما هو موضح أدناه:

```
PJRT_DEVICE=CUDA torchrun \
--nnodes=${NUMBER_GPU_VM} \
--node_rank=${CURRENT_NODE_RANK} \
--nproc_per_node=${NUMBER_LOCAL_GPU_DEVICES} \
--rdzv_endpoint=<internal_ip_address:port> multinode_training.py
```

- `--nnodes`: عدد أجهزة GPU التي سيتم استخدامها.
- `--node_rank`: فهرس أجهزة GPU الحالية. يمكن أن تكون القيمة 0 أو 1 أو ... أو ${NUMBER_GPU_VM}-1.
- `--nproc_per_node`: عدد أجهزة GPU التي سيتم استخدامها على الجهاز الحالي.
- `--rdzv_endpoint`: نقطة النهاية لجهاز GPU مع node_rank==0، على شكل `host:port`. سيكون `host` عنوان IP الداخلي. يمكن أن يكون `port` أي منفذ متاح على الجهاز. بالنسبة للتدريب/الاستدلال على عقدة واحدة، يمكن إغفال هذا المعلم.

على سبيل المثال، إذا كنت تريد التدريب على جهازي GPU: machine_0 وmachine_1، على جهاز GPU الأول machine_0، قم بتشغيل

```
# PJRT_DEVICE=CUDA torchrun \
--nnodes=2 \
--node_rank=0 \
--nproc_per_node=4 \
--rdzv_endpoint="<MACHINE_0_INTERNAL_IP_ADDRESS>:12355" pytorch/xla/test/test_train_mp_imagenet.py  --fake_data --pjrt_distributed --batch_size=128 --num_epochs=1
```

على جهاز GPU الثاني، قم بتشغيل

```
# PJRT_DEVICE=CUDA torchrun \
--nnodes=2 \
--node_rank=1 \
--nproc_per_node=4 \
--rdzv_endpoint="<MACHINE_0_INTERNAL_IP_ADDRESS>:12355" pytorch/xla/test/test_train_mp_imagenet.py  --fake_data --pjrt_distributed --batch_size=128 --num_epochs=1
```

الفرق بين الأمرين أعلاه هو `--node_rank` وربما `--nproc_per_node` إذا كنت تريد استخدام عدد مختلف من أجهزة GPU على كل جهاز. كل ما تبقى متطابق. لمزيد من المعلومات حول `torchrun`، يرجى الرجوع إلى هذه [الصفحة](https://pytorch.org/docs/stable/elastic/run.html).

## الاختلافات عن XRT

على الرغم من أنه في معظم الحالات نتوقع أن يعمل PJRT وXRT بشكل متبادل إلى حد كبير من منظور المستخدم النهائي (خاصة على TPU v4)، إلا أنه توجد بعض الاختلافات الدقيقة المهمة التي يجب مراعاتها. من المهم أن نلاحظ أن XRT تم تصميمه حول بنية عقدة TPU، لذلك سيقوم دائمًا بتشغيل عميل وعملية خادم، حتى على أجهزة TPU VM. وبالتالي، فإن كل دفعة من الإدخالات بها تأخير إضافي من تسلسل البيانات وإلغاء تسلسلها لإرسالها عبر الشبكة.

يستخدم PJRT الجهاز المحلي مباشرة دون عملية خادم وسيطة. في التكوين الافتراضي، سيقوم PJRT بإنشاء عملية واحدة لكل شريحة TPU، أو 4 عمليات لكل مضيف TPU. راجع [وثائق Cloud TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) لمزيد من المعلومات حول بنية TPU.

- مكاسب الأداء ممكنة للأحمال العملة المقيدة بسبب النفقات العامة من gRPC.
- في ظل XRT، تعد عملية الخادم هي العملية الوحيدة التي تتفاعل مع أجهزة TPU، ولا تملك عمليات العميل حق الوصول المباشر إلى أجهزة TPU.
عند إنشاء ملف تعريف لجهاز TPU واحد (مثل v3-8 أو v4-8)، فستلاحظ عادةً 8 مسارات جهاز (واحد لكل نواة TPU). مع PJRT، تحتوي كل عملية على شريحة واحدة، وسيعرض ملف التعريف من تلك العملية نواتي TPU فقط.
- لنفس السبب، لا يعمل التوصيل على مجموعات TPU مع XRT، لأن عملية الخادم تعمل بشكل مستقل عن رمز نموذج المستخدم. لا يملك PJRT هذا القيد، لذا فمن الممكن توصيل نواتي TPU لكل عملية في مجموعة TPU.
- يدعم PJRT فقط بنية VM TPU وليس لدينا خطط لدعم بنية عقدة TPU مع PJRT.
- تكوين وقت التشغيل أبسط بكثير مع PJRT. `xla_dist` غير مطلوب لتشغيل أحمال العمل على مجموعة TPU Pod. بدلاً من ذلك، قم بنسخ رمزك إلى كل مضيف TPU (`[gcloud compute tpus tpu-vm scp](https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/tpu-vm/scp)`) وقم بتشغيل الر
## التغييرات على xm.rendezvous 
_جديد في PyTorch/XLA r2.0_ 
مع XRT، يقوم العامل 0 بتشغيل خدمة رئيسية للشبكة، وتتصل جميع العمليات على جميع العاملين بتلك الخدمة عبر gRPC. وفي الممارسة العملية، وجدنا أن تشغيل عملية رئيسية واحدة للشبكة غير موثوق بها في وحدات TPU ذات الآلاف من الرقائق بسبب عدد الاتصالات الواردة إلى العامل 0. يمكن لعملية عميل واحدة توقيت يمكن أن يتسبب في حدوث فشل وإجبار عبء العمل بأكمله على إعادة التشغيل. 
لذلك، قمنا بإعادة تنفيذ `xm.rendezvous` باستخدام اتصال جماعي أصلي لـ XLA، وهو أكثر استقرارًا واختبارًا بشكل جيد على وحدات TPU الكبيرة. يفرض هذا قيدين جديدين مقارنة بتنفيذ XRT: 
* نظرًا لضرورة أن تصبح الحمولة جزءًا من رسم XLA، يتم استدعاء `xm.mark_step` قبل نقل البيانات وبعدها. قد يؤدي استدعاء `xm.rendezvous` في منتصف كود النموذج إلى فرض تجميع غير مرغوب فيه. 
* نظرًا لأن XLA لا تسمح بتشغيل العمليات الجماعية على مجموعة فرعية من العاملين، يجب على جميع العاملين المشاركة في "اللقاء". 
إذا كنت تحتاج إلى السلوك القديم لـ `xm.rendezvous` (أي نقل البيانات دون تغيير رسم XLA و/أو مزامنة مجموعة فرعية من العاملين)، ففكر في استخدام 
[`torch.distributed.barrier`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.barrier) 
أو 
[`torch.distributed.all_gather_object`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_object) 
مع مجموعة عمليات `gloo`. إذا كنت تستخدم أيضًا backend `xla` `torch.distributed`، فيمكنك استخدام `torch.new_group` لإنشاء مجموعة فرعية من نوع `gloo`. راجع [هذا المثال](https://pytorch.org/docs/stable/distributed.html#monitored-barrier) من وثائق PyTorch. ضع في اعتبارك هذه القيود: 
* `torch.distributed` غير مدعوم بالكامل على TPU v2/v3. يتم تنفيذ مجموعة فرعية فقط من العمليات مع backend `xla`، ومن المحتمل ألا يعمل `gloo` كما هو متوقع في سياق متعدد الخيوط. 
* في تجاربنا، لا يتوسع `gloo` جيدًا إلى الآلاف من رقائق TPU، لذا من المتوقع أن يكون هذا البديل أقل موثوقية من استخدام `xm.rendezvous` مع PJRT على نطاق واسع. 

## PJRT و torch.distributed 
_جديد في PyTorch/XLA r2.0_ 
عند استخدام PJRT مع `torch.distributed` و 
`[torch.nn.parallel.DistributedDataParallel](https://github.com/pytorch/xla/blob/master/docs/ddp.md)` 
نوصي بشدة باستخدام `xla://` `init_method` الجديد، والذي يقوم تلقائيًا باكتشاف معرفات النسخ المتماثلة وحجم العالم وعنوان IP الرئيسي عن طريق استعلام وقت التشغيل. على سبيل المثال: 

```python
import torch
import torch_xla
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt

# Required for `xla://` init_method and `xla` backend
import torch_xla.distributed.xla_backend

def _all_gather(index: int):
  # No need to pass in `rank` or `world_size`
  dist.init_process_group('xla', init_method='xla://')

  t = torch.tensor([index], dtype=torch.int32, device=xm.xla_device())
  output = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
  dist.all_gather(output, t)

  xm.mark_step()
  print(output)

if __name__ == '__main__':
  torch_xla.launch(_all_gather)
``` 
ملاحظة: على الرغم من أن `xla://` init_method غير مطلوب على TPU v4، إلا أنه لا يزال موصى به. إذا كنت تستخدم `env://`، فيجب تعيين `MASTER_ADDR` على عنوان IP المضيف الذي يحتوي على الجهاز 0، والذي ليس دائمًا العامل 0. تقوم طريقة init_method `xla://` بالعثور على هذا عنوان IP تلقائيًا. 
ملاحظة: بالنسبة لـ TPU v2/v3، لا يزال يتعين عليك استيراد `torch_xla.experimental.pjrt_backend`، حيث أن دعم TPU v2/v3 في `torch.distributed` لا يزال تجريبيًا. 
لمزيد من المعلومات حول استخدام `DistributedDataParallel` على PyTorch/XLA، راجع 
[`ddp.md`](./ddp.md) على TPU V4. للحصول على مثال يستخدم DDP و PJRT معًا، قم بتشغيل مثال البرنامج النصي التالي [example script](../test/test_train_mp_imagenet.py) على TPU: 
```
PJRT_DEVICE=TPU python xla/test/test_train_mp_mnist.py --ddp --pjrt_distributed --fake_data --num_epochs 1
``` 

## الأداء 
يظهر TorchBench تحسينات في متوسط وقت التدريب عبر المهام باستخدام PJRT مقارنة بـ XRT، مع متوسط تحسن يزيد عن 35% على TPU v4-8. تختلف الفوائد اختلافًا كبيرًا حسب المهمة ونوع النموذج، حيث تتراوح من 0% إلى 175%. يوضح الرسم البياني التالي التفاصك لكل مهمة: 
![PJRT مقابل XRT](_static/img/torchbench_pjrt_vs_xrt.svg) 

## وقت تشغيل TPU الجديد 
_جديد في PyTorch/XLA r2.0_ 
يقدم إصدار PyTorch/XLA r2.0 الدعم لـ [PJRT Plugin API](https://github.com/openxla/community/blob/main/rfcs/20230123-pjrt-plugin.md#rfc-openxla-pjrt-plugin)، المستخدم للوصول إلى وقت تشغيل TPU الجديد القائم على TFRT في `libtpu`. هذا هو وقت التشغيل الافتراضي الآن عند تعيين `PJRT_DEVICE=TPU`. سيظل وقت تشغيل TPU الموروث القائم على StreamExecutor المستخدم في الإصدار 1.13 متاحًا مع `PJRT_DEVICE=TPU_LEGACY` في الإصدار 2.0، ولكنه سيتم إزالته في إصدار مستقبلي. إذا واجهت مشكلة تحدث فقط على `TPU` وليس على `TPU_LEGACY`، يرجى إرسال تقرير عن المشكلة على GitHub. 
في معظم الحالات، نتوقع أن يكون الأداء مشابهًا بين وقتَي التشغيل، ولكن في بعض الحالات، قد يكون وقت التشغيل الجديد أسرع بنسبة تصل إلى 30%. يوضح الرسم البياني التالي التفاصيل حسب المهمة: 
![TFRT مقابل StreamExecutor](_static/img/torchbench_tfrt_vs_se.svg) 
ملاحظة: التحسينات الموضحة في هذا الرسم البياني مدرجة أيضًا في مقارنة PJRT مقابل XRT.