## Bazel في Pytorch/XLA

[Bazel](https://bazel.build/) هي أداة برمجية مجانية تستخدم لتشغيل بناء البرامج واختبارها. ويستخدمها كل من [TensorFlow](https://www.tensorflow.org/http) و [OpenXLA](https://github.com/openxla/xla)، مما يجعلها مناسبة أيضًا لـ PyTorch/XLA.

## تبعيات Bazel

تعد Tensorflow [تبعيات خارجية لـ Bazel](https://bazel.build/external/overview) لـ PyTorch/XLA، ويمكن رؤيتها في ملف `WORKSPACE`:

`WORKSPACE`

```bzl
http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-f7759359f8420d3ca7b9fd19493f2a01bd47b4ef",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/f7759359f8420d3ca7b9fd19493f2a01bd47b4ef.tar.gz",
    ],
)
```

يمكن تحديث تثبيت TensorFlow من خلال توجيه هذا المستودع إلى مراجعة مختلفة. ويمكن إضافة التصحيحات حسب الحاجة.

سيقوم Bazel بحل التبعية وإعداد الكود وتصحيحه بشكل محكم.

بالنسبة لـ PyTorch، يتم نشر آلية تبعية مختلفة لأننا نستخدم فحصًا محليًا لـ [PyTorch](https://github.com/pytorch/pytorch)، ويجب "بناء" هذا الفحص المحلي من المصدر وتثبيته في النظام من أجل توافق الإصدار (على سبيل المثال، يستخدم التوليد في PyTorch/XLA وحدة `torchgen` Python التي يجب تثبيتها في النظام).

يمكن تعيين الدليل المحلي إما في `bazel/dependencies.bzl`، أو تجاوزه في سطر الأوامر:

```bash
bazel build --override_repository=org_tensorflow=/path/to/exported/tf_repo //...
```

```bash
bazel build --override_repository=torch=/path/
to/exported/and/built/torch_repo //...
```

يرجى التأكد من أن المستودعات التي تم تجاوزها هي بالمراجع المناسبة، وفي حالة `torch`، تأكد من أنه قد تم بناؤه باستخدام `USE_CUDA=0 python setup.py bdist_wheel` للتأكد من وجود جميع كائنات البناء المتوقعة؛ يُفضل تثبيتها في النظام.

`WORKSPACE`

```bzl
new_local_repository(
    name = "torch",
    build_file = "//bazel:torch.BUILD",
    path = PYTORCH_LOCAL_DIR,
)
```

يتم الحصول على رؤوس PyTorch مباشرة من تبعية `torch`، وهي الفحص المحلي لـ PyTorch. يتم الحصول على المكتبات المشتركة (مثل `libtorch.so`) من نفس الفحص المحلي حيث تم بناء الكود ويحتوي `build/lib/` على الكائنات المبنية. لكي يعمل هذا، من الضروري تمرير `-isystemexternal/torch` إلى المترجم البرمجي حتى يتمكن من العثور على مكتبات "النظام" وتلبيتها من الفحص المحلي. يتم تضمين بعضها كمكتبات "<system>" والبعض الآخر كمكتبات `"user"` .

يجلب Bazel [pybind11](https://github.com/pybind/pybind11) Python المدمج ويربطه لتوفير `libpython` إلى المكون الإضافي باستخدام هذه الآلية. يتم أيضًا الحصول على رؤوس Python من هناك بدلاً من الاعتماد على إصدار النظام. يتم تلبيتها من `"@pybind11//:pybind11_embed"`، والذي يقوم بإعداد خيارات المترجم للربط مع `libpython` بشكل عابر.

## كيفية بناء مكتبات XLA

إن بناء المكتبات أمر بسيط:

```bash
bazel build //torch_xla/csrc/runtime/...
```

يمكن تكوين Bazel عبر `.bazelrc`، ولكنه يمكن أن يأخذ الأعلام أيضًا في سطر الأوامر.

```bash
bazel build --config=remote_cache //torch_xla/csrc/runtime/...
```

تستخدم تكوينات `remote_cache` gcloud للتخزين المؤقت وعادة ما تكون أسرع، ولكنها تتطلب المصادقة باستخدام gcloud. راجع `.bazelrc` للحصول على التكوين.

يجعل استخدام Bazel من السهل التعبير عن التبعيات المعقدة، وهناك الكثير من المكاسب من وجود رسم بياني بناء واحد حيث يتم التعبير عن كل شيء بنفس الطريقة. لذلك، لا توجد حاجة لبناء مكتبات XLA بشكل منفصل عن بقية المكون الإضافي كما كان الحال من قبل، حيث أن بناء المستودع بالكامل، أو الكائن المتشارك للمكون الإضافي الذي يربط كل شيء، هو أمر كافٍ.

## كيفية بناء مكون Torch/XLA الإضافي

يمكن تحقيق البناء العادي من خلال استدعاء `python setup.py bdist_wheel` القياسي، ولكن يمكن بناء روابط C++ ببساطة باستخدام:

```bash
bazel build //:_XLAC.so
```

سيقوم هذا ببناء عميل XLA والمكون الإضافي لـ PyTorch وربطهما معًا. يمكن أن يكون هذا مفيدًا عند اختبار التغييرات، ليكون قادرًا على تجميع كود C++ دون بناء المكون الإضافي لـ Python لدورات التكرار الأسرع.

## التخزين المؤقت البعيد

يأتي Bazel مع [التخزين المؤقت البعيد](https://bazel.build/remote/caching) مضمنًا. هناك العديد من واجهات التخزين المؤقت الخلفية التي يمكن استخدامها؛ نقوم بتنفيذ التخزين المؤقت لدينا على (GCS) [https://bazel.build/remote/caching#cloud-storage]. يمكنك الاطلاع على التكوين في `.bazelrc`، ضمن اسم التكوين `remote_cache`.

يتم تعطيل التخزين المؤقت البعيد بشكل افتراضي ولكن نظرًا لأنه يسرع عمليات البناء التزايدية بشكل كبير، فهو موصى به دائمًا تقريبًا، وهو مُمكّن بشكل افتراضي في أتمتة CI وعلى Cloud Build.

للمصادقة على جهاز ما، يرجى التأكد من وجود بيانات الاعتماد لديك باستخدام `gcloud auth application-default login --no-launch-browser` أو ما يعادلها.

يتطلب استخدام ذاكرة التخزين المؤقت البعيدة التي تم تكوينها بواسطة إعداد تكوين `remote_cache` المصادقة مع GCP.

هناك طرق مختلفة للمصادقة مع GCP. بالنسبة للمطورين الفرديين الذين لديهم حق الوصول إلى مشروع التطوير GCP، ما عليك سوى تحديد علم `--config=remote_cache` لـ Bazel، وسيتم استخدام `--google_default_credentials` الافتراضي وإذا كان رمز مصادقة gcloud موجودًا على الجهاز، فسيتم تشغيله خارج الصندوق، باستخدام المستخدم الذي تم تسجيل الدخول للتحقق من صحته. يجب أن يكون لدى المستخدم أذونات بناء بعيدة في GCP (أضف المطورين الجدد إلى دور "Bazel عن بُعد"). في CI، يتم استخدام مفتاح حساب الخدمة للمصادقة ويتم تمريره إلى Bazel باستخدام `--config=remote_cache --google_credentials=path/to/service.key`.

على [Cloud Build](https://cloud.google.com/build)، يتم استخدام `docker build --network=cloudbuild` لتمرير المصادقة من حساب الخدمة الذي يشغل Cloud Build إلى صورة Docker التي تقوم بالتجميع: تقوم [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc) بالعمل هناك ويتم المصادقة باستخدام حساب الخدمة. يجب أن يكون لدى جميع الحسابات، لكل من المستخدم وحسابات الخدمة، أذونات قراءة/كتابة للتخزين المؤقت البعيد.

يستخدم التخزين المؤقت البعيد صوامع التخزين المؤقت. يجب على كل آلة وبناء فريد تحديد مفتاح صومعة فريد للاستفادة من التخزين المؤقت المتسق. يمكن تمرير مفتاح الصومعة باستخدام علم: `-remote_default_exec_properties=cache-silo-key=SOME_SILO_KEY`.

تشغيل البناء باستخدام التخزين المؤقت البعيد:

```bash
BAZEL_REMOTE_CACHE=1 SILO_NAME="cache-silo-YOUR-USER" TPUVM_MODE=1 python setup.py bdist_wheel
```

قد يساعد أيضًا إضافة:

```bash
GCLOUD_SERVICE_KEY_FILE=~/.config/gcloud/application_default_credentials.json
```

إذا لم يتمكن "bazel" من العثور على رمز المصادقة.

يمكن أن يكون "YOUR-USER" هنا اسم مستخدم المؤلف أو اسم الجهاز، وهو اسم فريد يضمن سلوك التخزين المؤقت الجيد. تعمل وظائف "setup.py" الأخرى كما هو مقصود أيضًا (مثل "develop").

ستكون المرة الأولى التي يتم فيها تجميع الكود باستخدام مفتاح ذاكرة التخزين المؤقت الجديدة بطيئة لأنه سيقوم بتجميع كل شيء من الصفر، ولكن التجميعات التزايدية ستكون سريعة جدًا. عند تحديث تثبيت TensorFlow، سيكون الأمر أبطأ قليلاً في المرة الأولى لكل مفتاح، ثم حتى التحديث التالي سريعًا مرة أخرى.

## تشغيل الاختبارات

حاليًا، يقوم Bazel ببناء كود C++ واختباره. سيتم نقل كود Python في المستقبل.

Bazel هو أيضًا منصة اختبار، مما يسهل تشغيل الاختبارات:

```bash
bazel test //test/cpp:main
```

بالطبع، يجب أن يكون تكوين XLA وPJRT موجودًا في البيئة لتشغيل الاختبارات. لا يتم تمرير جميع المتغيرات البيئية إلى بيئة اختبار Bazel للتأكد من أن حالات عدم وجود ذاكرة التخزين المؤقت البعيدة ليست شائعة جدًا (البيئة جزء من مفتاح التخزين المؤقت)، راجع تكوين الاختبار `.bazelrc` لمعرفة تلك التي يتم تمريرها، وأضف الجديد حسب الحاجة.

يمكنك تشغيل الاختبارات باستخدام برنامج النصي المساعد أيضًا:

```bash
BAZEL_REMOTE_CACHE=1 SILO_NAME="cache-silo-YOUR-USER" ./test/cpp/run_tests.sh -R
```

اختبارات `xla_client` هي اختبارات محكمة بحتة يمكن تنفيذها بسهولة. اختبارات مكون `torch_xla` الإضافي أكثر تعقيدًا: فهي تتطلب تثبيت `torch` و`torch_xla`، ولا يمكن تشغيلها بالتوازي، لأنها تستخدم إما خادم/عميل XRT على نفس المنفذ، أو لأنها تستخدم جهاز GPU أو TPU ولا يتوفر سوى جهاز واحد في الوقت الحالي. لهذا السبب، يتم تجميع جميع الاختبارات الموجودة تحت `torch_xla/csrc/` في هدف واحد `:main` الذي يقوم بتشغيلها جميعًا بشكل تسلسلي.

## تغطية الكود

عند تشغيل الاختبارات، قد يكون من المفيد حساب تغطية الكود.

```bash
bazel coverage //torch_xla/csrc/runtime/...
```

يمكن تصور التغطية باستخدام `lcov` كما هو موضح في [توثيق Bazel](https://bazel.build/configure/coverage)، أو في محرر التعليمات البرمجية الخاص بك بمساعدة المكونات الإضافية لـ lcov، على سبيل المثال [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) لـ VSCode.

## خادم اللغة

يمكن لـ Bazel تشغيل خادم لغة مثل [clangd](https://clangd.llvm.org/) الذي يجلب المراجع البرمجية، والإكمال التلقائي، والفهم الدلالي للكود الأساسي إلى محرر التعليمات البرمجية الخاص بك. بالنسبة لـ VSCode، يمكن للمرء استخدام [Bazel Stack](https://github.com/stackb/bazel-stack-vscode-cc) الذي يمكن دمجه مع وظيفة [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) لجلب ميزات قوية لمساعدة تحرير الكود.

## بناء PyTorch/XLA

كما هو الحال دائمًا، يمكن بناء PyTorch/XLA باستخدام Python `distutils`:

```bash
BAZEL_REMOTE_CACHE=1 SILO_NAME="cache-silo-YOUR-USER" TPUVM_MODE=1 python setup.py bdist_wheel
```