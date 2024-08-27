## تكامل TorchDynamo (torch.compile) في PyTorch XLA

[TorchDynamo](https://pytorch.org/docs/stable/torch.compiler.html) هو مترجم JIT على مستوى Python مصمم لجعل برامج PyTorch غير المعدلة أسرع. فهو يوفر واجهة برمجة تطبيقات نظيفة لمؤخرات المترجم للربط بها، وتتمثل ميزته الأكبر في تعديل بايت كود Python ديناميكيًا مباشرة قبل تنفيذه. في إصدار pytorch/xla 2.0، قدم PyTorch/XLA مؤخرًا تجريبية لـ TorchDynamo لكل من الاستدلال والتدريب.

طريقة عمل جسر XLA هي أن Dynamo سيوفر رسم TorchFX عندما يتعرف على نمط نموذج، وسيستخدم PyTorch/XLA تقنية Tensor الكسول الحالية لتجميع رسم FX وإرجاع الدالة المجمعة.

### التكامل

يتم دعم PyTorch/XLA وDynamo حاليًا عن طريق إضافة وسيط `backend='openxla'` إلى `torch.compile`. على سبيل المثال:

```py
import torch
import torch_xla.core.xla_model as xm

def add(a, b):
  a_xla = a.to(xm.xla_device())
  b_xla = b.to(xm.xla_device())
  return a_xla + b_xla

compiled_code = torch.compile(add, backend='openxla')
print(compiled_code(torch.randn(10), torch.randn(10)))
```

### الاستدلال

فيما يلي مثال صغير على تشغيل resnet18 مع `torch.compile`

```python
import torch
import torchvision
import torch_xla.core.xla_model as xm

def eval_model(loader):
  device = xm.xla_device()
  xla_resnet18 = torchvision.models.resnet18().to(device)
  xla_resnet18.eval()
  dynamo_resnet18 = torch.compile(
    xla_resnet18, backend='openxla')
  for data, _ in loader:
    with torch.no_grad():
      output = dynamo_resnet18(data)
```

مع `torch.compile`، ستلاحظ أن PyTorch/XLA يقوم بتتبع نموذج resent18 مرة واحدة فقط أثناء وقت التشغيل وتنفيذ التعليمات البرمجية المجمعة في كل مرة يتم فيها استدعاء `dynamo_resnet18`، بدلاً من تتبع النموذج في كل مرة. فيما يلي تحليل لسرعة الاستدلال لمقارنة Dynamo وLazy باستخدام مقعد الشعلة على Cloud TPU v4-8

| النموذج | تسريع |
| --- | ----------- |
resnet18 | 2.59
resnet50 | 2.64
resnext50_32x4d| 1.91
alexnet | 1.28
mobilenet_v2 | 18.62
mnasnet1_0 | 2.68
vgg16 | 1.33
BERT_pytorch | 7.49
squeezenet1_1 | 2.29
timm_vision_transformer | 3.52
geomean | 3.04

### التدريب

يدعم PyTorch/XLA أيضًا Dynamo للتدريب، ولكنه تجريبي ونحن نعمل مع فريق PyTorch Compiler لتحسين التنفيذ. فيما يلي مثال على تدريب resnet18 مع `torch.compile`

```python
import torch
import torchvision
import torch_xla.core.xla_model as xm

def train_model(model, data, target, optimizer):
  loss_fn = torch.nn.CrossEntropyLoss()
  pred = model(data)
  loss = loss_fn(pred, target)
  loss.backward()
  optimizer.step()
  return pred

def train_model_main(loader):
  device = xm.xla_device()
  xla_resnet18 = torchvision.models.resnet18().to(device)
  xla_resnet18.train()
  dynamo_train_model = torch.compile(
        train_model, backend='openxla')
  for data, target in loader:
    xla_optimizer = optim.SGD(data, lr=0.1, weight_decay=1e-2)
    output = dynamo_train_model(xla_resnet18, data, target, xla_optimizer)
```

نتوقع استخراج وتنفيذ 3 رسومات لكل خطوة تدريب بدلاً من رسم بياني واحد لكل خطوة تدريب إذا كنت تستخدم Tensor الكسول. فيما يلي تحليل لسرعة التدريب لمقارنة Dynamo وLazy باستخدام مقعد الشعلة على Cloud TPU v4-8.

| النموذج | تسريع |
| --- | ----------- |
resnet50 | 1.33
resnet18 | 1.33
BERT_pytorch | 3.07
resnext50_32x4d | 1.43
alexnet | 1.12
mobilenet_v2 | 1.4
mnasnet1_0 | 1.19
vgg16 | 0.81
timm_vision_transformer | 1.87
squeezenet1_1 | 1.41
geomean | 1.41

> **ملاحظة:** نقوم بتشغيل fwd وbwd لكل نموذج لخطوة واحدة ثم نجمع وقت e2e. في العالم الحقيقي، سنقوم بتشغيل خطوات متعددة في كل مهمة تدريب يمكن أن تخفي بسهولة تكلفة التتبع من التنفيذ (بما أنه غير متزامر). سيكون Tensor الكسول أداء أفضل بكثير في هذا السيناريو.

### الفجوات الميزة

هناك فجوة واحدة نريد أن نوضحها تمنعنا من استخدام TorchDynamo على نماذج أكبر.

1. سيقوم TorchDynamo بتتبع الخطوات إلى الأمام والخلف في رسومات منفصلة. بالنسبة لـ PyTorch/XLA، من المهم السماح لمترجم XLA برؤية الخطوة الكاملة كرسم بياني واحد لتحسين السرعة بشكل أفضل. هناك أيضًا تكلفة ثابتة لإطلاق كل تنفيذ جهاز، مما يجعل تنفيذ رسومات متعددة لكل خطوة تدريب أقل مثالية.

تجعل هذه الفجوة مقارنة بـ Tensor الكسول أقل كفاءة في حالات الاستخدام التدريبية في العالم الحقيقي، خاصة أن تكلفة التتبع يمكن أن تتداخل مع التنفيذ في التدريب.

### خلاصة

يوفر TorchDynamo طريقة واعدة لمؤخرات المترجم لإخفاء التعقيد عن المستخدم واسترداد رمز النمذجة بسهولة بتنسيق الرسم البياني. مقارنة بطريقة Tensor الكسول التقليدية لـ PyTorch/XLA لاستخراج الرسم البياني، يمكن لـ TorchDynamo تخطي تتبع الرسم البياني لكل تكرار، مما يوفر وقت استجابة أفضل للاستدلال.

شهدت معظم النماذج التي يدعمها PyTorch/XLA تسريعًا كبيرًا عند تشغيل الاستدلال بجسر dynamo-xla الجديد. يعمل مجتمعنا بجد لتوسيع مجموعة النماذج المدعومة. فيما يتعلق بفجوات ميزات التدريب المذكورة أعلاه، فإن مجتمع PyTorch/XLA متحمس للغاية لتحسين فجوة التدريب في عمل التطوير القادم لدينا. يواصل الفريق الاستثمار بكثافة في TorchDynamo والعمل مع الجهات الخارجية لتحسين قصة التدريب.