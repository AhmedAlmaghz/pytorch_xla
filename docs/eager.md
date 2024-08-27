## وضع Eager + واجهة برمجة تطبيقات Compile

في هذه الوثيقة، سنشرح كيفية استخدام الوضع `Eager` التجريبي الجديد في PyTorch/XLA مع واجهة برمجة التطبيقات `Compile`. الهدف من ذلك هو جعل تجربة PyTorch/XLA أكثر توافقاً مع PyTorch الأصلي، وتسهيل عملية التطوير.

## الخلفية

يعمل PyTorch/XLA حاليًا في وضع تتبع LazyTensor بشكل افتراضي. في الكود التالي:

```python
import torch
import torch_xla
import torchvision

device = torch_xla.device()
model = torchvision.models.resnet18().to(device)
input = torch.randn(64, 3, 224, 224).to(device)

# تتبع النموذج
res = model(input)

# تنفيذ النموذج، نفس xm.mark_step
torch_xla.sync()
```

يحدث التجميع الفعلي للنموذج وتنفيذ الجهاز عند استدعاء `torch_xla.sync`. هناك العديد من العيوب لهذا النهج:

1. غالبًا ما يكون المستخدمون مرتبكين بشأن متى يقوم الإطار بالتتبع ومتى يقوم بالتنفيذ.
2. غالبًا ما تقوم التعليمات البرمجية غير الأساسية للنموذج (مثل المعالجة المسبقة للبيانات) بتوليد بعض التنفيذات المعلقة الصغيرة التي تتسرب إلى الرسم البياني الرئيسي (دالة الخطوة) وتتسبب في إعادة التجميع. وعادة ما تكون إعادة تجميع الرسم البياني بالكامل مكلفة للغاية.
3. من الصعب تصحيح الأخطاء عندما/لماذا تحدث إعادة التجميع.

للتغلب على هذه المشكلات، نريد تقديم تجربة المستخدم الجديدة مع `Eager` و `Compile`.

## الاستخدام الأساسي

```python
import torch
import torch_xla
import torchvision

# تشغيل العمليات في الوضع Eager بشكل افتراضي
torch_xla.experimental.eager_mode(True)

device = torch_xla.device()
model = torchvision.models.resnet18().to(device)

# وضع علامة على الدالة ليتم تجميعها
compiled_model = torch_xla.compile(model)
input = torch.randn(64, 3, 224, 224).to(device)

# يحدث التجميع والتنفيذ على الفور.
res = compiled_model(input)
```

يرجى ملاحظة ما يلي:

1. يجب على المستخدم حاليًا تمكين الوضع `Eager` يدويًا عن طريق `torch_xla.experimental.eager_mode(True)`.
2. يجب لف الجزء من الكود الذي نريد تجميعه باستخدام `torch_xla.compile`.

إن تنفيذ `torch_xla.compile` مباشر إلى حد ما، حيث يقوم بتعطيل الوضع `Eager` عند الدخول إلى الدالة الهدف ويبدأ في التتبع. وسوف يستدعي `torch_xla.sync()` عندما تعيد الدالة الهدف تمكين الوضع `Eager` مرة أخرى. يمكنك توقع نفس الأداء عند استخدام واجهة برمجة التطبيقات `Eager` + `Compile` مقارنة بنهج `mark_step/sync` الحالي.

### الاستنتاج

```python
torch_xla.experimental.eager_mode(True)

compiled_model = torch.compile(model, backend="openxla")
```

من المستحسن استخدام `torch.compile` بدلاً من `torch_xla.compile` للاستدلال لتقليل عبء التتبع.

### التدريب

```python
torch_xla.experimental.eager_mode(True)

def step_fn(model, data, target, loss_fn, optimizer):
    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    return loss

step_fn = torch_xla.compile(step_fn)
```

في التدريب، طلبنا من المستخدم إعادة هيكلة `step_fn` لأنه من الأفضل عادة تجميع عملية التقديم للأمام والخلف والمحسن للنموذج معًا. والهدف على المدى الطويل هو أيضًا استخدام `torch.compile` للتدريب، ولكن في الوقت الحالي نوصي المستخدم باستخدام `torch_xla.compile` (لأسباب تتعلق بالأداء).

## المعيار المرجعي

قمت بتشغيل نموذج فك تشفير مكون من طبقتين فقط (وهو في الأساس Llama2) مع بيانات وهمية على شريحة واحدة من v4-8 لمدة 300 خطوة. وفيما يلي الأرقام التي لاحظتها:

<table>
<tr>
<td>
</td>
<td>الرموز المميزة/ثانية
</tr>
<tr>
<td>وضع التتبع (خط الأساس)
</td>
<td>147
</td>
</td>
</tr>
<tr>
<td>وضع Eager
</td>
<td>65
</td>
</td>
</tr>
<tr>
<td>Eager + تجميع torch_xla
</td>
<td>147
</td>
</td>
</tr>
</table>

يمكن أن يحقق وضع `Eager` حوالي 45% من أداء النموذج المجمع بالكامل لنموذج فك التشفير فقط. ويمكن العثور على المدرب الذي استخدمته للاختبار [هنا](https://github.com/pytorch/xla/blob/master/examples/train_decoder_only_base.py) و [هنا](https://github.com/pytorch/xla/tree/master/examples/eager). يرجى ملاحظة أن أداء وضع `Eager` يعتمد بشدة على النموذج. فعندما حاولت تشغيل `ResNet50`، كان أداء وضع `Eager` حوالي 1% من وضع `Compiled`. لا نتوقع من المستخدم استخدام وضع `Eager` لتنفيذ حلقة التدريب الرئيسية. إنما الغرض من وضع `Eager` هو التعامل مع الجزء غير الأساسي من منطق التدريب/الاستدلال (مثل المعالجة المسبقة للبيانات، وتوليد الأرقام العشوائية، وما إلى ذلك) أو التصحيح.