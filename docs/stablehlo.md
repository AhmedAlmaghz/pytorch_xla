# التحويل من Torch إلى StableHLO

يصف هذا المستند كيفية استخدام التصدير الشعلة + الشعلة xla للتصدير إلى تنسيق [StableHLO](https://github.com/openxla/stablehlo).

## كيفية الاستخدام:

```python
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo
import torch_xla.core.xla_model as xm
import torchvision
import torch

xla_device = xm.xla_device()

resnet18 = torchvision.models.resnet18()
# Sample input is a tuple
sample_input = (torch.randn(4, 3, 224, 224), )
output = resnet18(*sample_input)
exported = export(resnet18, sample_input)
stablehlo_program = exported_program_to_stablehlo(exported)

# Now stablehlo_program is a callable backed by stablehlo IR.

# we can see it's stablehlo code with
# Here 'forward' is the name of the function. Currently we only support
# one entry point per program, but in the future we will support
# multiple entry points in a program.
print(stablehlo_program.get_stablehlo_text('forward'))

# we can also print out the bytecode
print(stablehlo_program.get_stablehlo_bytecode('forward'))

# we can also run the module, to run the stablehlo module, we need to move
# our tensors to XLA device.
sample_input_xla = tuple(s.to(xla_device) for s in sample_input)

output2 = stablehlo_program(*sample_input_xla)
print(torch.allclose(output, output2.cpu(), atol=1e-5))
```

# حفظ تعليمات StableHLO البرمجية على القرص:

يمكنك الآن حفظ stablehlo على القرص باستخدام:

```python
stablehlo_program.save('/tmp/stablehlo_dir')
```

يجب أن يكون المسار هو المسار إلى دليل فارغ. إذا لم يكن موجودًا، فسيتم إنشاؤه.

يمكن تحميل هذا الدليل مرة أخرى كـ stablehlo_program آخر:

```python
from torch_xla.stablehlo import StableHLOGraphModule
stablehlo_program2 = StableHLOGraphModule.load('/tmp/stablehlo_dir')
output3 = stablehlo_program2(*sample_input_xla)
```

# تحويل StableHLO المحفوظة للخدمة:

StableHLO هو تنسيق مفتوح ومدعوم للخدمة في خادم نموذج [tensorflow.serving](https://github.com/tensorflow/serving). ومع ذلك، قبل تقديمه إلى tf.serving، يلزم أولاً تغليف تعليمات StableHLO البرمجية المولدة في تنسيق `tf.saved_model`.

للقيام بذلك، تأكد أولاً من تثبيت أحدث إصدار من tensorflow في بيئة Python الحالية، وإذا لم يكن الأمر كذلك، فقم بالتثبيت باستخدام:

```bash
pip install tf-nightly
```

الآن، يمكنك تشغيل محول (مقدم في تثبيت torch/xla)

```
stablehlo-to-saved-model /tmp/stablehlo_dir /tmp/resnet_tf/1
```

بعد ذلك، يمكنك تشغيل خادم النموذج على "tf.saved_model" الذي تم إنشاؤه حديثًا باستخدام ثنائي الخدمة tf.

```
docker pull tensorflow/serving
docker run -p 8500:8500 \
--mount type=bind,source=/tmp/resnet_tf,target=/models/resnet_tf \
-e MODEL_NAME=resnet_tf -t tensorflow/serving &
```

يمكنك أيضًا استخدام ثنائي "tf.serving" مباشرة دون الحاجة إلى Docker.

للحصول على مزيد من التفاصيل، يرجى اتباع [دليل الخدمة tf](https://www.tensorflow.org/tfx/serving/serving_basic).

# الأغطية الشائعة:

### أريد حفظ تنسيق "tf.saved_model" مباشرةً دون الحاجة إلى تشغيل أمر منفصل.

يمكنك تحقيق ذلك باستخدام دالة المساعدة هذه:

```python
from torch_xla.tf_saved_model_integration import save_torch_module_as_tf_saved_model

save_torch_module_as_tf_saved_model(
    resnet18,  # original pytorch torch.nn.Module 
    sample_inputs, # sample inputs used to trace
    '/tmp/resnet_tf'   # directory for tf.saved_model
)
```

### الأغطية الشائعة الأخرى:

```python
def save_as_stablehlo(exported_model: 'ExportedProgram',
                      stablehlo_dir: os.PathLike,
                      options: Optional[StableHLOExportOptions] = None):
```

`save_as_stablehlo` (أيضًا مستعار باسم `torch_xla.save_as_stablehlo`)
يأخذ ExportedProgram ويحفظ StableHLO على القرص. أي
نفس exported_program_to_stablehlo(...).save(...)

```python
def save_torch_model_as_stablehlo(
    torchmodel: torch.nn.Module,
    args: Tuple[Any],
    path: os.PathLike,
    options: Optional[StableHLOExportOptions] = None) -> None:
  """Convert a torch model to a callable backed by StableHLO.

```
يأخذ `torch.nn.Module` ويحفظ StableHLO على القرص. أي
نفس export.export الشعلة تليها save_as_stablehlo

# الملفات التي ينتجها `save_as_stablehlo`.

داخل `/tmp/stablehlo_dir` في المثال أعلاه، ستجد 3 مجلدات: `data`، `constants`، `functions`. وسيحتوي كل من البيانات والثوابت على المصفوفات المستخدمة بواسطة البرنامج
تم حفظه كـ `numpy.ndarray` باستخدام `numpy.save`.

سيتضمن دليل الوظائف تعليمات StableHLO البرمجية، والتي تحمل هنا اسم `forward.bytecode`، ورمز StableHLO القابل للقراءة البشري (شكل MLIR) `forward.mlir`، وملف JSON يحدد الأوزان
وإدخالات المستخدم الأصلية تصبح الحجج الموضعية لهذه الدالة StableHLO؛ وكذلك
أنواع البيانات والأشكال لكل حجة.

مثال:

```
$ find /tmp/stablehlo_dir
./functions
./functions/forward.mlir
./functions/forward.bytecode
./functions/forward.meta
./constants
./constants/3
./constants/1
./constants/0
./constants/2
./data
./data/L__fn___layers_15_feed_forward_w2.weight
./data/L__fn___layers_13_feed_forward_w1.weight
./data/L__fn___layers_3_attention_wo.weight
./data/L__fn___layers_12_ffn_norm_weight
./data/L__fn___layers_25_attention_wo.weight
...
```

ملف JSON هو الشكل التسلسلي لفئة `torch_xla.stablehlo.StableHLOFunc`.

هذا التنسيق هو أيضًا في مرحلة النموذج الأولي حاليًا ولا توجد ضمانات للتوافق مع الإصدارات السابقة.
وتتمثل الخطة المستقبلية في توحيد تنسيق يمكن للإطارات الرئيسية (PyTorch وJAX وTensorFlow) الاتفاق عليه.

# الحفاظ على عمليات PyTorch عالية المستوى في StableHLO عن طريق إنشاء `stablehlo.composite`

سيتم تحليل عمليات PyTorch عالية المستوى (مثل `F.scaled_dot_product_attention`) إلى عمليات منخفضة المستوى أثناء خفض مستوى PyTorch -> StableHLO. يمكن أن يكون التقاط العملية عالية المستوى في برامج التجميع ML أسفل البنية أمرًا بالغ الأهمية لإنشاء نوى متخصصة فعالة وفعالة. في حين أن مطابقة مجموعة من العمليات منخفضة المستوى في برنامج التجميع ML يمكن أن يكون أمرًا صعبًا وعرضة للأخطاء، فإننا نقدم طريقة أكثر متانة لتحديد موقع العملية عالية المستوى في برنامج StableHLO - عن طريق إنشاء [stablehlo.composite](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#composite) لعمليات PyTorch عالية المستوى.

مع `StableHLOCompositeBuilder`، يمكن للمستخدم تحديد منطقة تعسفية داخل وظيفة `forward` لـ `torch.nn.Module`. بعد ذلك، في برنامج StableHLO الذي تم تصديره، سيتم إنتاج عملية مركبة للمنطقة المحددة.

**ملاحظة:** نظرًا لأن قيمة الإدخالات غير المصفوفة للمنطقة المحددة ستكون ثابتة في الرسم البياني الذي تم تصديره، يرجى تخزين تلك القيم كسمات مركبة، إذا كان الاسترداد من برنامج التجميع أسفل البنية مطلوبًا.

يوضح المثال التالي حالة استخدام عملية - التقاط `scaled_product_attention`

```python
import torch
import torch.nn.functional as F
from torch_xla import stablehlo
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder


class M(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(128, 128, bias=False)
        self.k_proj = torch.nn.Linear(128, 128, bias=False)
        self.v_proj = torch.nn.Linear(128, 128, bias=False)
        # Initialize the StableHLOCompositeBuilder with the name of the composite op and its attributes
        # Note: To capture the value of non-tensor inputs, please pass them as attributes to the builder
        self.b = StableHLOCompositeBuilder("test.sdpa", {"scale": 0.25, "other_attr": "val"})

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k, v = self.b.mark_inputs(q, k, v)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
        attn_out = self.b.mark_outputs(attn_out)
        attn_out = attn_out + x
        return attn_out

input_args = (torch.randn((10, 8, 128)), )
# torch.export to Exported Program
exported = torch.export.export(M(), input_args)
# Exported Program to StableHLO
stablehlo_gm = stablehlo.exported_program_to_stablehlo(exported)
stablehlo = stablehlo_gm.get_stablehlo_text()
print(stablehlo)
```

يتم عرض الرسم البياني StableHLO الرئيسي أدناه:

```mlir
module @IrToHlo.56 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<10x8x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>) -> tensor<10x8x128xf32> {
    ...
    %10 = stablehlo.composite "test.sdpa" %3, %6, %9 {composite_attributes = {other_attr = "val", scale = 2.500000e-01 : f32}, decomposition = @test.sdpa.impl} : (tensor<10x8x128xf32>, tensor<10x8x128xf32>, tensor<10x8x128xf32>) -> tensor<10x8x128xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<10x8x128xf32>
    return %11 : tensor<10x8x128xf32>
  }

  func.func private @test.sdpa.impl(%arg0: tensor<10x8x128xf32>, %arg1: tensor<10x8x128xf32>, %arg2: tensor<10x8x128xf32>) -> tensor<10x8x128xf32> {
    // Actual implementation of the composite
    ...
    return %11 : tensor<10x8x128xf32>
  }
```

يتم تغليف عملية sdpa كدعوة مركبة stablehlo داخل الرسم البياني الرئيسي. يتم نقل الاسم والسمات المحددة في وحدة "torch.nn"

```mlir
%10 = stablehlo.composite "test.sdpa" %3, %6, %9 {composite_attributes = {other_attr = "val", scale = 2.500000e-01 : f32}, decomposition = @test.sdpa.impl}
```

يتم التقاط تفكيك PyTorch المرجعي لعملية sdpa في وظيفة StableHLO:

```mlir
func.func private @test.sdpa.impl(%arg0: tensor<10x8x128xf32>, %arg1: tensor<10x8x128xf32>, %arg2: tensor<10x8x128xf32>) -> tensor<10x8x128xf32> {
    // Actual implementation of the composite
    ...
    return %11 : tensor<10x8x128xf32>
  }
```