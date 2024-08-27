# OpenXLA 

اعتبارًا من 28 يونيو 2023، يقوم PyTorch/XLA الآن بسحب XLA من OpenXLA.
OpenXLA هو [مُجمِّع مصدر مفتوح XLA لتعلم الآلة لمسرعات GPUs وCPUs وML](https://github.com/openxla/xla).

قبل OpenXLA، كان PyTorch/XLA يسحب XLA مباشرةً من [TensorFlow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla). مع [هجرة XLA إلى OpenXLA](https://github.com/pytorch/xla/pull/5202)، يقوم PyTorch/XLA الآن بسحب XLA من [OpenXLA](https://github.com/openxla/xla).

# كيفية استخدام OpenXLA

بالنسبة لمستخدمي [وقت تشغيل PJRT](https://github.com/pytorch/xla/blob/master/docs/pjrt.md)، لا يوجد تغيير مع هذا الانتقال. وبالنسبة لمستخدمي وقت تشغيل XRT، هناك فرع منفصل لـ [XRT من PyTorch/XLA](https://github.com/pytorch/xla/tree/xrt) نظرًا لأن OpenXLA لا يدعم XRT.

# الأداء

فيما يلي مقارنة مرئية للأداء بين الإنتاجية قبل وبعد الهجرة على أجهزة TPU المختلفة.

|  | resnet50-pjrt-v2-8 | resnet50-pjrt-v4-8 | resnet50-pjrt-v4-32 |
| :------------ | :------------ | :------------ | :------------ |
| قبل الهجرة | 18.59 | 20.06 | 27.92 |
| بعد الهجرة | 18.63 | 19.94 | 27.14 |