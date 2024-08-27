## Fully Sharded Data Parallel (FSDP) في PyTorch XLA

تعتبر أداة Fully Sharded Data Parallel (FSDP) في PyTorch XLA أداة مفيدة لتقسيم معاملات الوحدات عبر العمّال المتوازيين للبيانات.

مثال على الاستخدام:

```python3
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

model = FSDP(my_module)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
output = model(x, y)
loss = output.sum()
loss.backward()
optim.step()
```

من الممكن أيضًا تقسيم الطبقات الفردية بشكل منفصل وإحاطة أي معلمات متبقية بمُحيط خارجي.

ملاحظات:

* تدعم فئة `XlaFullyShardedDataParallel` كل من محسن ZeRO-2 (تقسيم المعاملات وتدرجات الحالة) ومحسن ZeRO-3 (تقسيم المعاملات والتدرجات وحالات المحسن) في [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054).

* يجب تنفيذ محسن ZeRO-3 من خلال FSDP متداخلة مع `reshard_after_forward=True`. راجع `test/test_train_mp_mnist_fsdp_with_ckpt.py` و`test/test_train_mp_imagenet_fsdp.py` للحصول على مثال.

* بالنسبة للنماذج الكبيرة التي لا يمكن أن تتناسب مع ذاكرة TPU واحدة أو ذاكرة وحدة المعالجة المركزية المضيفة، يجب أن تتشابك عملية بناء الوحدة الفرعية مع الإحاطة الداخلية لـ FSDP. راجع [`FSDPViTModel`](https://github.com/ronghanghu/vit_10b_fsdp_example/blob/master/run_vit_training.py) للحصول على مثال.

* يتم توفير غلاف بسيط `checkpoint_module` (بناءً على `torch_xla.utils.checkpoint.checkpoint` من https://github.com/pytorch/xla/pull/3524) لأداء [التحقق من نقطة التدرج](https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs) على مثيل `nn.Module` معين. راجع `test/test_train_mp_mnist_fsdp_with_ckpt.py` و`test/test_train_mp_imagenet_fsdp.py` للحصول على مثال.

* الإحاطة التلقائية بالوحدات الفرعية: بدلاً من الإحاطة اليدوية لـ FSDP المتداخلة، يمكنك أيضًا تحديد حجة `auto_wrap_policy` لإحاطة الوحدات الفرعية تلقائيًا بـ FSDP الداخلية. `size_based_auto_wrap_policy` في `torch_xla.distributed.fsdp.wrap` هو مثال على `auto_wrap_policy` قابل للاستدعاء، حيث يقوم هذا النهج بإحاطة الطبقات بعدد من المعلمات أكبر من 100 مليون. `transformer_auto_wrap_policy` في `torch_xla.distributed.fsdp.wrap` هو مثال على `auto_wrap_policy` قابل للاستدعاء لهندسات النماذج الشبيهة بالمحول.

على سبيل المثال، لإحاطة جميع الوحدات الفرعية `torch.nn.Conv2d` تلقائيًا بـ FSDP الداخلية، يمكنك استخدام ما يلي:

```python3
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={torch.nn.Conv2d})
```

بالإضافة إلى ذلك، يمكنك أيضًا تحديد حجة `auto_wrapper_callable` لاستخدام وظيفة استدعاء مخصصة لإحاطة الوحدات الفرعية (غلاف الافتراضي هو فئة `XlaFullyShardedDataParallel` نفسها). على سبيل المثال، يمكنك استخدام ما يلي لتطبيق التحقق من التدرج (أي التحقق من نقطة التنشيط/إعادة المادة) على كل وحدة فرعية ملفوفة تلقائيًا.

```python3
from torch_xla.distributed.fsdp import checkpoint_module
auto_wrapper_callable = lambda m, *args, **kwargs: XlaFullyShardedDataParallel(
  checkpoint_module(m), *args, **kwargs)
```

* عند تشغيل المحسن، اتصل مباشرة بـ `optimizer.step` ولا تتصل بـ `xm.optimizer_step`. يؤدي هذا الأخير إلى تقليل التدرج عبر الرتب، وهو غير مطلوب لـ FSDP (حيث يتم تقسيم المعلمات بالفعل).

* عند حفظ نقاط التحقق للنموذج والمحسن أثناء التدريب، يجب على كل عملية تدريب حفظ نقطة التحقق الخاصة بها من القواميس (المقسمة) لحالة النموذج والمحسن (استخدم `master_only=False` وحدد مسارات مختلفة لكل رتبة في `xm.save`). عند الاستئناف، يجب تحميل نقطة التحقق للرتبة المقابلة.

* يرجى أيضًا حفظ `model.get_shard_metadata()` جنبًا إلى جنب مع `model.state_dict()` كما هو موضح أدناه واستخدام `consolidate_sharded_model_checkpoints` لربط نقاط التحقق للنموذج المقسم معًا في قاموس حالة النموذج الكامل. راجع `test/test_train_mp_mnist_fsdp_with_ckpt.py` للحصول على مثال.

```python3
ckpt = {
  'model': model.state_dict(),
  'shard_metadata': model.get_shard_metadata(),
  'optimizer': optimizer.state_dict(),
}
ckpt_path = f'/tmp/rank-{xr.global_ordinal()}-of-{xr.world_size()}.pth'
xm.save(ckpt, ckpt_path, master_only=False)
```

* يمكن أيضًا تشغيل برنامج توحيد نقاط التحقق من سطر الأوامر كما هو موضح أدناه.

```bash
# دمج نقاط التحقق المحفوظة عبر أداة سطر الأوامر
python3 -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts \
  --ckpt_prefix /path/to/your_sharded_checkpoint_files \
  --ckpt_suffix "_rank-*-of-*.pth"
```

يستلهم تنفيذ هذه الفئة إلى حد كبير من هيكل `fairscale.nn.FullyShardedDataParallel` في https://fairscale.readthedocs.io/en/stable/api/nn/fsdp.html. أحد أكبر الاختلافات عن `fairscale.nn.FullyShardedDataParallel` هو أنه في XLA، لا يوجد لدينا تخزين معلمات صريح، لذا فإننا نلجأ إلى نهج مختلف لتحرير المعلمات الكاملة لـ ZeRO-3.

---

### مثال على البرامج النصية للتدريب على MNIST وImageNet

* الحد الأدنى من الأمثلة: [`examples/fsdp/train_resnet_fsdp_auto_wrap.py`](https://github.com/pytorch/xla/blob/master/examples/fsdp/train_resnet_fsdp_auto_wrap.py)

* MNIST: [`test/test_train_mp_mnist_fsdp_with_ckpt.py`](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_fsdp_with_ckpt.py) (يختبر أيضًا توحيد نقاط التحقق)

* ImageNet: [`test/test_train_mp_imagenet_fsdp.py`](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_fsdp.py)

#### التثبيت

تتوفر FSDP على PyTorch/XLA 1.12 release والإصدارات الأحدث من nightly. يرجى الرجوع إلى https://github.com/pytorch/xla#-available-images-and-wheels للحصول على دليل التثبيت.

#### استنساخ مستودع PyTorch/XLA

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
git clone --recursive https://github.com/pytorch/xla.git
cd ~/
```

#### تدريب MNIST على v3-8 TPU

يحصل على دقة تبلغ حوالي 98.9 لدورتين:

```bash
python3 ~/pytorch/xla/test/test_train_mp_mnist_fsdp_with_ckpt.py \
  --batch_size 16 --drop_last --num_epochs 2 \
  --use_nested_fsdp --use_gradient_checkpointing
```

يختبر هذا البرنامج النصي تلقائيًا توحيد نقاط التحقق في النهاية. يمكنك أيضًا توحيد نقاط التحقق المقسمة يدويًا كما هو موضح أدناه:

```bash
# دمج نقاط التحقق المحفوظة عبر أداة سطر الأوامر
python3 -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts \
  --ckpt_prefix /tmp/mnist-fsdp/final_ckpt \
  --ckpt_suffix "_rank-*-of-*.pth"
```

#### تدريب ImageNet مع ResNet-50 على v3-8 TPU

يحصل على دقة تبلغ حوالي 75.9 لـ 100 حقبة؛ قم بتنزيل [ImageNet-1k](https://github.com/pytorch/examples/tree/master/imagenet#requirements) إلى `/datasets/imagenet-1k`:

```bash
python3 ~/pytorch/xla/test/test_train_mp_imagenet_fsdp.py \
  --datadir /datasets/imagenet-1k --drop_last \
  --model resnet50 --test_set_batch_size 64 --eval_interval 10 \
  --lr 0.4 --batch_size 128 --num_warmup_epochs 5 --lr_scheduler_divide_every_n_epochs 30 \
  --lr_scheduler_divisor 10 --num_epochs 100 \
  --use_nested_fsdp
```
يمكنك أيضًا إضافة ` --use_gradient_checkpointing` (الذي يجب أن يستخدم مع `--use_nested_fsdp` أو `--auto_wrap_policy`) لتطبيق نقاط التحقق التدريجية على الكتل المتبقية.

---

### نصوص تدريب مثالية على حزمة TPU (مع 10 مليارات معلمة)

لتدريب نماذج كبيرة لا تتسع في TPU واحد، يجب تطبيق التغليف التلقائي أو تغليف الوحدات الفرعية يدويًا باستخدام FSDP الداخلي عند بناء النموذج بأكمله لتنفيذ خوارزمية ZeRO-3.

يرجى الاطلاع على https://github.com/ronghanghu/vit_10b_fsdp_example للحصول على مثال عن التدريب المجزأ لنموذج Vision Transformer (ViT) باستخدام هذا الإصدار من XLA FSDP.
