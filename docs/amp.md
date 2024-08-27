# AMP (Automatic Mixed Precision) with Pytorch/XLA

توسع AMP من Pytorch/XLA في حزمة AMP من Pytorch بدعم الدقة المختلطة التلقائية على أجهزة XLA: GPU و XLA: TPU.

يستخدم AMP لتسريع التدريب والاستدلال عن طريق تنفيذ عمليات معينة في "float32" وعمليات أخرى في نوع بيانات أقل دقة ("float16" أو "bfloat16" حسب دعم الأجهزة).

تصف هذه الوثيقة كيفية استخدام AMP على أجهزة XLA وأفضل الممارسات.

## AMP for XLA:TPU

يقوم AMP على وحدات معالجة Tensor تلقائيًا بتحويل العمليات لتشغيلها إما في "float32" أو "bfloat16" لأن وحدات معالجة Tensor تدعم bfloat16 بشكل أصلي. وفيما يلي مثال بسيط على AMP لوحدة معالجة Tensor:

```py
# Creates model and optimizer in default precision
model = Net().to(xm.xla_device())
# Pytorch/XLA provides sync-free optimizers for improved performance
optimizer = syncfree.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass
    with autocast(xm.xla_device()):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    xm.optimizer_step.(optimizer)
```

`autocast(xm.xla_device())` عبارة عن اسم مستعار لـ `torch.autocast('xla')` عندما يكون جهاز XLA عبارة عن وحدة معالجة Tensor. أو، إذا كان البرنامج النصي يستخدم فقط مع وحدات معالجة Tensor، فيمكن استخدام `torch.autocast('xla', dtype=torch.bfloat16)` مباشرة.

يرجى تقديم طلب سحب أو إرسال طلب إذا كان هناك عامل تشغيل يجب أن يكون autocasted غير مدرج.

### أفضل الممارسات

1. يجب أن يقوم `autocast` بتغليف تمريرات للأمام وحسابات الخسارة للشبكة فقط. تعمل عمليات الخلف في نفس النوع الذي استخدمه autocast للتمريرات الأمامية المقابلة.
2. نظرًا لأن وحدات معالجة Tensor تستخدم الدقة المختلطة bfloat16، فإن تدرج التدرج غير ضروري.
3. يوفر Pytorch/XLA إصدارًا معدلًا من المحسنات التي تتجنب المزامنة الإضافية بين الجهاز والمضيف.

### المشغلون المدعومون

يعمل AMP على وحدات معالجة Tensor مثل AMP من Pytorch. فيما يلي ملخص لقواعد كيفية تطبيق التحويل التلقائي:

- لا يحق إلا للعمليات غير الموضعية وطرق Tensor أن تكون autocasted. ويُسمح بالمتغيرات الموضعية والمكالمات التي توفر صراحةً Tensor خارجيًا في المناطق التي تم تمكين autocast فيها، ولكنها لن تخضع للتحويل التلقائي. على سبيل المثال، في منطقة ممكنة autocast، يمكن أن يكون a.addmm(b، c) autocasted، ولكن لا يمكن أن يكون a.addmm_(b، c) و a.addmm(b، c، out=d). للحصول على أفضل أداء واستقرار، يفضل استخدام العمليات غير الموضعية في المناطق الممكنة autocast.
- العمليات التي تعمل في float64 أو الأنواع غير العائمة غير مؤهلة، وستعمل في هذه الأنواع سواء تم تمكين autocast أم لا. بالإضافة إلى ذلك، فإن العمليات التي يتم استدعاؤها باستخدام حجة dtype=... صريحة غير مؤهلة، وستنتج ناتجًا يحترم حجة dtype.
- لا تخضع العمليات غير المدرجة أدناه للتحويل التلقائي. إنها تعمل في النوع الذي تحدده إدخالاتها. قد يؤدي التحويل التلقائي لا يزال إلى تغيير النوع الذي تعمل فيه العمليات غير المدرجة إذا كانت أسفل العمليات التي تم تحويلها تلقائيًا.

**المشغلون الذين يقومون بالتحويل التلقائي إلى `bfloat16`:**

`__matmul__`، `addbmm`، `addmm`، `addmv`، `addr`، `baddbmm`، `bmm`، `conv1d`، `conv2d`، `conv3d`، `conv_transpose1d`، `conv_transpose2d`، `conv_transpose3d`، `linear`، `matmul`، `mm`، `relu`، `prelu`، `max_pool2d`.

**المشغلون الذين يقومون بالتحويل التلقائي إلى `float32`:**

`batch_norm`، `log_softmax`، `binary_cross_entropy`، `binary_cross_entropy_with_logits`، `prod`، `cdist`، `trace`، `cholesky`، `inverse`، `reflection_pad`، `replication_pad`، `mse_loss`، `cosine_embedding_loss`، `nll_loss`، `multilabel_margin_loss`، `qr`، `svd`، `triangular_solve`، `linalg_svd`، `linalg_inv_ex`.

**المشغلون الذين يقومون بالتحويل التلقائي إلى نوع الإدخال الأوسع:**

`stack`، `cat`، `index_copy`.

## AMP for XLA:GPU

يعيد AMP على أجهزة XLA: GPU استخدام قواعد AMP من Pytorch. راجع وثائق AMP من Pytorch للسلوك المحدد لـ CUDA. فيما يلي مثال بسيط على AMP لـ CUDA:

```py
# Creates model and optimizer in default precision
model = Net().to(xm.xla_device())
# Pytorch/XLA provides sync-free optimizers for improved performance
optimizer = syncfree.SGD(model.parameters(), ...)
scaler = GradScaler()

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass
    with autocast(xm.xla_device()):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward pass
    scaler.scale(loss).backward()
    gradients = xm._fetch_gradients(optimizer)
    xm.all_reduce('sum', gradients, scale=1.0 / xr.world_size())
    scaler.step(optimizer)
    scaler.update()
```

`autocast(xm.xla_device())` عبارة عن اسم مستعار لـ `torch.cuda.amp.autocast()` عندما يكون جهاز XLA عبارة عن جهاز CUDA (XLA: GPU). أو، إذا كان البرنامج النصي يستخدم فقط مع أجهزة CUDA، فيمكن استخدام `torch.cuda.amp.autocast` مباشرةً، ولكنه يتطلب دعم `torch` مع دعم `cuda` لنوع بيانات `torch.bfloat16`. نوصي باستخدام `autocast(xm.xla_device())` على XLA: GPU لأنه لا يتطلب دعم `torch.cuda` لأي أنواع بيانات، بما في ذلك `torch.bfloat16`.

### أفضل الممارسات

1. يجب أن يقوم `autocast` بتغليف تمريرات للأمام وحسابات الخسارة للشبكة فقط. تعمل عمليات الخلف في نفس النوع الذي استخدمه autocast للتمريرات الأمامية المقابلة.
2. لا تقم بتعيين علم `XLA_USE_F16` عند استخدام AMP على أجهزة Cuda. سيؤدي هذا إلى تجاوز إعدادات الدقة لكل مشغل المقدمة بواسطة AMP والتسبب في تنفيذ جميع المشغلين في float16.
3. استخدم تدرج التدرج لمنع تدرجات float16 من الانخفاض.
4. يوفر Pytorch/XLA إصدارًا معدلًا من المحسنات التي تتجنب المزامنة الإضافية بين الجهاز والمضيف.

## أمثلة

يوضح [برنامج نصي لتدريب mnist](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_amp.py) و [برنامج نصي لتدريب imagenet](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_amp.py) لدينا كيفية استخدام AMP على وحدات معالجة Tensor ووحدات معالجة الرسوميات.