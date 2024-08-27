# الحفظ الموزع للإشارات المرجعية

يتوافق PyTorch/XLA SPMD مع مكتبة [torch.distributed.checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html) من خلال مثيل مخصص لـ `Planner`. يمكن للمستخدمين حفظ الأحمال المرجعية وتحميلها بشكل متزامن من خلال هذا الواجهة العامة.

تمكّن فئات SPMDSavePlanner و SPMDLoadPlanner ([src](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/distributed_checkpoint.py)) وظائف "الحفظ" و"التحميل" من العمل مباشرة على شرائح XLAShardedTensor، مما يتيح جميع مزايا الحفظ الموزع للإشارات المرجعية في التدريب SPMD.

فيما يلي توضيح لواجهة برمجة التطبيقات الموزعة المتزامنة للإشارات المرجعية:

```python
import torch.distributed.checkpoint as dist_cp
import torch_xla.experimental.distributed_checkpoint as xc

# Saving a state_dict
state_dict = {
    "model": model.state_dict(),
    "optim": optim.state_dict(),
}

dist_cp.save(
    state_dict=state_dict,
    storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
    planner=xc.SPMDSavePlanner(),
)
...

# Loading the model's state_dict from the checkpoint. The model should
# already be on the XLA device and have the desired sharding applied.
state_dict = {
    "model": model.state_dict(),
}

dist_cp.load(
    state_dict=state_dict,
    storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
    planner=xc.SPMDLoadPlanner(),
)
model.load_state_dict(state_dict["model"])
```

#### CheckpointManager

توفر واجهة [CheckpointManager](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/distributed_checkpoint/manager.py#L40) التجريبية واجهة برمجة تطبيقات أعلى مستوى من وظائف `torch.distributed.checkpoint` لتمكين بعض الميزات الرئيسية:

- **الإشارات المرجعية المُدارة**: يتم تحديد كل إشارة مرجعية يتم أخذها بواسطة CheckpointManager
من خلال الخطوة التي تم اتخاذها. يمكن الوصول إلى جميع الخطوات التي يتم تتبعها
من خلال طريقة CheckpointManager.all_steps، ويمكن استعادة أي خطوات يتم تتبعها
باستخدام CheckpointManager.restore.

- **الحفظ الموزع غير المتزامر**: يتم كتابة الإشارات المرجعية المتخذة من خلال واجهة برمجة تطبيقات CheckpointManager.save_async بشكل غير متزامر إلى التخزين الدائم لإلغاء حظر التدريب أثناء الإشارة المرجعية. يتم أولاً نقل القاموس المُجزء إلى وحدة المعالجة المركزية قبل إرسال الإشارة المرجعية إلى مؤشر ترابط في الخلفية.

- **الحفظ التلقائي عند الاستيلاء**: يمكن اكتشاف عمليات الاستيلاء على Cloud TPU واتخاذ إشارة مرجعية قبل إنهاء العملية. لاستخدام هذه الميزة، تأكد من تخصيص وحدة TPU الخاصة بك من خلال QueuedResource مع [تمكين الحفظ التلقائي](https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/queued-resources/create#--autocheckpoint-enabled)، وتأكد من تعيين معلمة chkpt_on_preemption عند إنشاء CheckpointManager (يكون هذا الخيار ممكّنًا بشكل افتراضي).

- **دعم FSSpec**: يستخدم CheckpointManager backend للتخزين المستند إلى FSSpec لتمكين الحفظ المباشر إلى أي نظام ملفات متوافق مع FSSpec، بما في ذلك GCS.

مثال على استخدام CheckpointManager موضح أدناه:

```python
from torch_xla.experimental.distributed_checkpoint import CheckpointManager, prime_optimizer

# Create a CheckpointManager to checkpoint every 10 steps into GCS.
chkpt_mgr = CheckpointManager('gs://my-bucket/my-experiment', 10)

# Select a checkpoint to restore from, and restore if applicable
tracked_steps = chkpt_mgr.all_steps()
if tracked_steps:
    # Choose the highest step
    best_step = max(tracked_steps)
    # Before restoring the checkpoint, the optimizer state must be primed
    # to allow state to be loaded into it.
    prime_optimizer(optim)
    state_dict = {'model': model.state_dict(), 'optim': optim.state_dict()}
    chkpt_mgr.restore(best_step, state_dict)
    model.load_state_dict(state_dict['model'])
    optim.load_state_dict(state_dict['optim'])

# Call `save` or `save_async` every step within the train loop. These methods
# return True when a checkpoint is taken.
for step, data in enumerate(dataloader):
    ...
    state_dict = {'model': model.state_dict(), 'optim': optim.state_dict()}
    if chkpt_mgr.save_async(step, state_dict):
        print(f'Checkpoint taken at step {step}')
```

##### استعادة حالة المحسن

في الحفظ الموزع للإشارات المرجعية، يتم تحميل القواميس الحالة في المكان، ويتم تحميل الشرائح المطلوبة فقط من الإشارة المرجعية. نظرًا لأن حالات المحسن يتم إنشاؤها بشكل مؤجل، فإن الحالة لا تكون موجودة حتى يتم إجراء أول مكالمة `optimizer.step`، وستفشل محاولات تحميل المحسن غير الممهد.

يتم توفير طريقة utility `prime_optimizer` لهذا الغرض: فهي تقوم بتشغيل خطوة تدريب وهمية عن طريق تعيين جميع التدرجات إلى الصفر واستدعاء `optimizer.step`. *هذه طريقة مدمرة وستؤثر على كل من معلمات النموذج وحالة المحسن*، لذا يجب استدعاؤها فقط قبل الاستعادة مباشرة.

### مجموعات العمليات

لاستخدام واجهات برمجة التطبيقات الموزعة لـ `torch.distributed` مثل الحفظ الموزع للإشارات المرجعية، مطلوب مجموعة عمليات. في وضع SPMD، لا يتم دعم backend "xla" لأن المترجم مسؤول عن جميع العمليات الجماعية.

بدلاً من ذلك، يجب استخدام مجموعة عمليات وحدة المعالجة المركزية مثل "gloo". في وحدات TPU، لا يزال init_method "xla://" مدعومًا لاكتشاف عنوان IP الرئيسي وحجم العالم العالمي والترتيب المضيف. موضح أدناه مثال على التهيئة:

```python
import torch.distributed as dist
# استيراد لتسجيل init_method "xla://"
import torch_xla.distributed.xla_backend
import torch_xla.runtime as xr

xr.use_spmd()

# ستكتشف طريقة init_method "xla://" تلقائيًا عنوان IP الرئيسي للعمال وحجم العالم العالمي والترتيب
# دون الحاجة إلى تكوين البيئة على وحدات TPU.
dist.init_process_group('gloo', init_method='xla://')
```