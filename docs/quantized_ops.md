# عمليات كمية لأجهزة XLA (ميزة تجريبية)
--------------------------
توضح هذه الوثيقة كيفية استخدام العمليات الكمية لتمكين التكميم على أجهزة XLA.
توفر عمليات XLA الكمية مستوى عالي من التجريد للعمليات الكمية (مثل، الضرب الكمي للمصفوفات blockwise int4). هذه العمليات مماثلة لنوى CUDA الكمية ([مثال](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/gptq/q_gemm.cu)) في نظام CUDA، وتوفر وظائف ومزايا أداء مماثلة داخل إطار عمل XLA.
**ملاحظة:** حالياً، يتم تصنيف هذه الميزة على أنها تجريبية. سيتم تغيير تفاصيل واجهة برمجة التطبيقات الخاصة بها في الإصدار التالي (2.5).

## كيفية الاستخدام:
يمكن استخدام العمليات الكمية XLA كـ `torch op`، أو كـ `torch.nn.Module` الذي يغلف `torch.op`. يمنح هذان الخياران مطورو النماذج المرونة لاختيار أفضل طريقة لدمج العمليات الكمية XLA في حلولهم.
يتوافق كل من `torch op` و`nn.Module` مع `torch.compile( backend='openxla')`.

### استدعاء العملية الكمية XLA في كود النموذج
يمكن للمستخدمين استدعاء العمليات الكمية XLA بنفس طريقة استدعاء عمليات PyTorch العادية الأخرى. يوفر ذلك أقصى قدر من المرونة في دمج العمليات الكمية XLA في تطبيقاتهم. تعمل العمليات الكمية في كل من الوضع الفوري وDynamo، مع مصفوفة PyTorch CPU العادية ومصفوفة XLA.
**ملاحظة** يرجى التحقق من docstring للعمليات الكمية لمعرفة تخطيط الأوزان الكمية.
```Python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_quantized_matmul

N_INPUT_FEATURES=10
N_OUTPUT_FEATURES=20
x = torch.randn((3, N_INPUT_FEATURES), dtype=torch.bfloat16)
w_int = torch.randint(-128, 127, (N_OUTPUT_FEATURES, N_INPUT_FEATURES), dtype=torch.int8)
scaler = torch.randn((N_OUTPUT_FEATURES,), dtype=torch.bfloat16)

# الاستدعاء باستخدام مصفوفة CPU الخاصة بـ PyTorch (لغرض التصحيح)
matmul_output = torch.ops.xla.quantized_matmul(x, w_int, scaler)

device = xm.xla_device()
x_xla = x.to(device)
w_int_xla = w_int.to(device)
scaler_xla = scaler.to(device)

# الاستدعاء باستخدام مصفوفة XLA لتشغيلها على جهاز XLA
matmul_output_xla = torch.ops.xla.quantized_matmul(x_xla, w_int_xla, scaler_xla)

# الاستخدام مع torch.compile(backend='openxla')
def f(x, w, s):
  return torch.ops.xla.quantized_matmul(x, w, s)

f_dynamo = torch.compile(f, backend="openxla")
dynamo_out_xla = f_dynamo(x_xla, w_int_xla, scaler_xla)
```
من الشائع تغليف العملية الكمية في وحدة `nn.Module` مخصصة في كود نموذج المطور:
```Python
class MyQLinearForXLABackend(torch.nn.Module):
  def __init__(self):
    self.weight = ...
    self.scaler = ...

  def load_weight(self, w, scaler):
    # Load quantized Linear weights
    # Customized way to preprocess the weights
    ...
    self.weight = processed_w
    self.scaler = processed_scaler

  
  def forward(self, x):
    # Do some random stuff with x
    ...
    matmul_output = torch.ops.xla.quantized_matmul(x, self.weight, self.scaler)
    # Do some random stuff with matmul_output
    ...
```

### تبديل الوحدات
بدلاً من ذلك، يمكن للمستخدمين أيضًا استخدام وحدة `nn.Module` التي تغلف العمليات الكمية XLA وإجراء تبديل الوحدات في كود النموذج:

```Python
orig_model = MyModel()
# تكميم النموذج والحصول على الأوزان الكمية
q_weights = quantize(orig_model)
# معالجة الأوزان الكمية لتتوافق مع تنسيق العملية الكمية XLA.
q_weights_for_xla = process_for_xla(q_weights)

# إجراء تبديل الوحدات
q_linear = XlaQuantizedLinear(self.linear.in_features,self.linear.out_features)
q_linear.load_quantized_weight(q_weights_for_xla)
orig_model.linear = q_linear
```

## العمليات الكمية المدعومة:
### ضرب المصفوفات
| نوع تكميم الأوزان | نوع تكميم التنبيهات | Dtype | مدعوم |
|---|---|---|---|
| لكل قناة (sym/asym) | N/A | W8A16 | نعم |
| لكل قناة (sym/asym) | N/A | W4A16 | نعم |
| لكل قناة | لكل رمز | W8A8 | لا |
| لكل قناة | لكل رمز | W4A8 | لا |
| blockwise (sym/asym) | N/A | W8A16 | نعم |
| blockwise (sym/asym) | N/A | W4A16 | نعم |
| blockwise | لكل رمز | W8A8 | لا |
| blockwise | لكل رمز | W4A8 | لا |
**ملاحظة** تشير `W[X]A[Y]` إلى وزن في `X`-بت، وتنشيط في `Y`-بت. إذا كان `X/Y` هو 4 أو 8، فإنه يشير إلى `int4/8`. 16 لتنسيق `bfloat16`.

### تضمين
سيتم إضافته