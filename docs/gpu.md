# كيفية التشغيل باستخدام PyTorch/XLA:GPU 

يتيح PyTorch/XLA لمستخدمي PyTorch الاستفادة من مترجم XLA الذي يدعم المعالجات المُسرعة بما في ذلك TPU و GPU و CPU. ستتناول هذه الوثيقة الخطوات الأساسية لتشغيل PyTorch/XLA على مثيلات GPU من Nvidia.

## إنشاء مثيل GPU

يمكنك استخدام جهاز محلي متصل به GPU أو آلة افتراضية GPU على السحابة. على سبيل المثال، في Google Cloud، يمكنك اتباع هذه [الوثيقة](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus) لإنشاء آلة افتراضية GPU.

## إعداد البيئة

تأكد من تثبيت برنامج تشغيل CUDA على المضيف.

### Docker

ينشر PyTorch/XLA حاليًا صور Docker مُسبقة البناء وعجلات مع cuda11.8/12.1 و Python 3.8. نوصي المستخدمين بإنشاء حاوية Docker مع التكوين المقابل. للحصول على قائمة كاملة بصور Docker والعجلات، يُرجى الرجوع إلى [هذه الوثيقة](https://github.com/pytorch/xla#available-docker-images-and-wheels).

```
sudo docker pull us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1

# Installing the NVIDIA Container Toolkit per https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# For example
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configuring the NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo docker run --shm-size=16g --net=host --gpus all -it -d us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1 bin/bash
sudo docker exec -it $(sudo docker ps | awk 'NR==2 { print $1 }') /bin/bash
```

ملاحظة: يلزمك إعادة تشغيل Docker لجعل أجهزة GPU مرئية في حاوية Docker. بعد تسجيل الدخول إلى Docker، يمكنك استخدام `nvidia-smi` للتحقق من إعداد الجهاز بشكل صحيح.

```
(pytorch) root@20ab2c7a2d06:/# nvidia-smi
Thu Dec  8 06:24:29 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   36C    P0    38W / 300W |      0MiB / 16384MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### التحقق من متغير البيئة

تأكد من أن متغيرات البيئة `PATH` و `LD_LIBRARY_PATH` تأخذ في الاعتبار cuda. يرجى تشغيل `echo $PATH` و `echo $LD_LIBRARY_PATH` للتحقق. إذا لم يكن الأمر كذلك، يرجى اتباع [هذا الرابط](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#mandatory-actions) للقيام بذلك. مثال:

```
echo "export PATH=\$PATH:/usr/local/cuda-12.1/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64" >> ~/.bashrc
source ~/.bashrc
```

### Wheel

> **_ملاحظة:_** ملف العجلة متوافق فقط مع بنية x86_64 Linux. للتحقق من بنية نظام Linux الخاص بك، نفذ الأمر التالي:
> ```
> uname -a
> ```

```
pip3 install torch==2.4.0
# GPU whl for python 3.10 + cuda 12.1
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl
```

يمكن العثور على العجلات الخاصة بإصدارات Python و CUDA الأخرى [هنا](https://github.com/pytorch/xla?tab=readme-ov-file#available-docker-images-and-wheels).

## تشغيل بعض النماذج البسيطة

لتشغيل الأمثلة أدناه، تحتاج إلى استنساخ مستودع pytorch/xla.

### مثال MP_ImageNet

يستخدم هذا المثال ImageNet. وهو مدرج في ما قمنا باستنساخه بالفعل في حاوية Docker الخاصة بنا.

```
(pytorch) root@20ab2c7a2d06:/# export GPU_NUM_DEVICES=1 PJRT_DEVICE=CUDA
(pytorch) root@20ab2c7a2d06:/# git clone --recursive https://github.com/pytorch/xla.git
(pytorch) root@20ab2c7a2d06:/# python xla/test/test_train_mp_imagenet.py --fake_data
==> Preparing data..
Epoch 1 train begin 06:12:38
| Training Device=xla:0/0 Epoch=1 Step=0 Loss=6.89059 Rate=2.82 GlobalRate=2.82 Time=06:13:23
| Training Device=xla:0/0 Epoch=1 Step=20 Loss=6.79297 Rate=117.16 GlobalRate=45.84 Time=06:13:36
| Training Device=xla:0/0 Epoch=1 Step=40 Loss=6.43628 Rate=281.16 GlobalRate=80.49 Time=06:13:43
| Training Device=xla:0/0 Epoch=1 Step=60 Loss=5.83108 Rate=346.88 GlobalRate=108.82 Time=06:13:49
| Training Device=xla:0/0 Epoch=1 Step=80 Loss=4.99023 Rate=373.62 GlobalRate=132.43 Time=06:13:56
| Training Device=xla:0/0 Epoch=1 Step=100 Loss=3.92699 Rate=384.33 GlobalRate=152.40 Time=06:14:02
| Training Device=xla:0/0 Epoch=1 Step=120 Loss=2.68816 Rate=388.35 GlobalRate=169.49 Time=06:14:09
```

### مثال ResNet

يستخدم هذا المثال ResNet.

```
(pytorch) root@20ab2c7a2d06:/# python3 /xla/examples/train_resnet_base.py
1:35PM UTC on Jun 08, 2024
epoch: 1, step: 0, loss: 6.887794017791748, rate: 8.746502586051985
epoch: 1, step: 10, loss: 6.877807140350342, rate: 238.4789458412044
epoch: 1, step: 20, loss: 6.867819786071777, rate: 329.86095958663503
epoch: 1, step: 30, loss: 6.857839584350586, rate: 367.3038003653586
epoch: 1, step: 40, loss: 6.847847938537598, rate: 381.53141087190835
epoch: 1, step: 50, loss: 6.837860584259033, rate: 387.80462249591113
...
epoch: 1, step: 260, loss: 6.628140926361084, rate: 391.135639565343
epoch: 1, step: 270, loss: 6.618192195892334, rate: 391.6901797745233
epoch: 1, step: 280, loss: 6.608224391937256, rate: 391.1602680460045
epoch: 1, step: 290, loss: 6.598264217376709, rate: 391.6731498290759
Epoch 1 train end  1:36PM UTC
```

## AMP (الدقة المختلطة التلقائية)

تعد AMP مفيدة جدًا في تدريب GPU ويعيد PyTorch/XLA استخدام قاعدة AMP من Cuda. يمكنك الاطلاع على مثالنا [mnist example](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_amp.py) و [imagenet example](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_amp.py). لاحظ أننا استخدمنا أيضًا إصدارًا معدلًا من [optimizers](https://github.com/pytorch/xla/tree/master/torch_xla/amp/syncfree) لتجنب المزامنة الإضافية بين الجهاز والمضيف.

## تطوير PyTorch/XLA على مثيل GPU (قم ببناء PyTorch/XLA من المصدر مع دعم GPU)

1. داخل آلة افتراضية GPU، قم بإنشاء حاوية Docker من صورة Docker للتطوير. على سبيل المثال:

```
sudo docker pull us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:3.8_cuda_12.1

# Installing the NVIDIA Container Toolkit per https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# For example
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configuring the NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo docker run --shm-size=16g --net=host --gpus all -it -d us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:3.8_cuda_12.1
sudo docker exec -it $(sudo docker ps | awk 'NR==2 { print $1 }') /bin/bash
```

2. قم ببناء PyTorch و PyTorch/XLA من المصدر.

تأكد من أن متغيرات البيئة `PATH` و `LD_LIBRARY_PATH` تأخذ في الاعتبار cuda. راجع [الأعلى](https://github.com/pytorch/xla/blob/master/docs/gpu.md#check-environment-variable) للحصول على مزيد من المعلومات.

```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
USE_CUDA=1 python setup.py install
USE_CUDA=1 python setup.py bdist_wheel # مطلوب لـ Python المحكم في إعداد البناء PyTorch/XLA.

git clone https://github.com/pytorch/xla.git
cd xla
XLA_CUDA=1 python setup.py install
```

3. تحقق مما إذا كان PyTorch و PyTorch/XLA قد تم تثبيتهما بنجاح.

إذا تمكنت من تشغيل الاختبارات في القسم [Run some simple models](#run-some-simple-models) بنجاح، فيجب تثبيت PyTorch و PyTorch/XLA بنجاح.