# The Cloud TPU Workflow

هدف هذا الدليل هو إعداد بيئة تطوير تفاعلية على
Cloud TPU مع PyTorch/XLA المثبت. إذا كانت هذه هي المرة الأولى التي تستخدم فيها TPUs، فإننا
نوصي ببدء استخدام
[Colab](https://colab.sandbox.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb)
و [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338) أو.
كلتا الخيارين لديها PyTorch/XLA مثبتة مسبقًا مع التبعيات وحزم النظام البيئي
الحزم. للحصول على قائمة محدثة من الأمثلة، راجع قائمة [README](https://github.com/pytorch/xla) الرئيسية لدينا.

إذا كنت ترغب في إعداد بيئة تطوير أكثر تخصيصًا، فاستمر في القراءة.

## Visual Studio Code

المتطلبات الأساسية:

- [Visual Studio Code](https://code.visualstudio.com/download) مع [Remote
  Development
  extensions](https://code.visualstudio.com/docs/remote/remote-overview)
  المثبتة على جهازك المحلي
- مشروع GCP مع حصة Cloud TPU. للحصول على مزيد من المعلومات حول طلب
  حصة Cloud TPU، راجع الوثائق [الرسمية](https://cloud.google.com/tpu/docs/quota).
- مفتاح SSH مسجل مع `ssh-agent`. إذا لم تقم بذلك بالفعل، فراجع وثائق
  [GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

قبل البدء، قم بتصدير متغيرات البيئة مع مشروع GCP والمنطقة
حيث لديك حصة Cloud TPU:

```
export PROJECT=...
export ZONE=...
export TPU_TYPE=... # e.g. "v2-8"
```

### إنشاء والاتصال بجهاز TPU الخاص بك

قم بإنشاء جهاز TPU VM مع مفتاح SSH الخاص بك مسجل:

```bash
# افتراض أن مفتاح SSH الخاص بك يسمى `id_ed25519`
gcloud compute tpus tpu-vm create --project=$PROJECT --zone=$ZONE --accelerator-type=$TPU_TYPE --version=tpu-ubuntu2204-base --metadata="ssh-keys=$USER:$(cat ~/.ssh/id_ed25519.pub)" $USER-tpu
```

تحقق من أن جهاز TPU الخاص بك لديه عنوان IP خارجي واتصال SSH به:

```bash
gcloud compute tpus tpu-vm describe --project=$PROJECT --zone=$ZONE $USER-tpu --format="value(networkEndpoints.accessConfig.externalIp)"
# Output: 123.123.123.123
```

أعطِ جهاز TPU الخاص بك اسمًا ودودًا لتسهيل الخطوات المستقبلية:

```bash
echo -e Host $USER-tpu "\n " HostName $(gcloud compute tpus tpu-vm describe --project=$PROJECT --zone=$ZONE $USER-tpu --format="value(networkEndpoints.accessConfig.externalIp)") >> ~/.ssh/config
```

اتصال SSH بجهاز TPU الخاص بك لاختبار الاتصال:

```
ssh $USER-tpu
```

### إعداد مساحة عمل Visual Studio Code مع PyTorch/XLA

من [VS Code Command
Palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette)،
حدد [`Remote-SSH: Connect to
Host`](https://code.visualstudio.com/docs/remote/ssh) وحدد المضيف الذي أنشأته للتو (يسمى `$USER-tpu`). بعد ذلك، سيقوم VS Code بفتح نافذة جديدة متصلة
بجهاز TPU VM الخاص بك.

من المحطة الطرفية المدمجة، قم بإنشاء مجلد جديد لاستخدامه كمساحة عمل (على سبيل المثال
`mkdir ptxla`). بعد ذلك، افتح المجلد من واجهة المستخدم أو شريط الأوامر.

ملاحظة: من الاختياري (ولكن يوصى به) في هذه المرحلة تثبيت الإضافية الرسمية
[Python
extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
وإنشاء [`venv` virtual
environment](https://code.visualstudio.com/docs/python/environments#_using-the-create-environment-command)
عبر شريط الأوامر (`Python: Create Environment`).

قم بتثبيت أحدث إصدارات PyTorch و PyTorch/XLA:

```
pip install numpy torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

قم بإنشاء ملف `test.py`:

```python
import torch_xla as xla

# Optional
xla.runtime.set_device_type("TPU")

print("XLA devices:", xla.real_devices())
```

قم بتشغيل البرنامج النصي للاختبار من المحطة الطرفية الخاصة بك:

```bash
$ python test.py
# Output: XLA devices: ['TPU:0', 'TPU:1', 'TPU:2', 'TPU:3', 'TPU:4', 'TPU:5', 'TPU:6', 'TPU:7']
# سيختلف عدد الأجهزة بناءً على نوع TPU
```

### الخطوات التالية

هذا كل شيء! يجب أن يكون لديك الآن مساحة عمل Visual Studio Code عن بُعد تم إعدادها مع
تم تثبيت PyTorch/XLA. لتشغيل أمثلة أكثر واقعية، راجع دليل [الأمثلة](https://github.com/pytorch/xla/tree/master/examples) الخاص بنا.