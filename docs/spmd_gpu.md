# تشغيل SPMD على وحدة معالجة الرسومات (GPU)

يدعم PyTorch/XLA تشغيل SPMD على وحدة معالجة الرسومات NVIDIA (عقدة واحدة أو متعددة). ويظل نص البرمجة النصية للتدريب/الاستدلال كما هو المستخدم لوحدة معالجة tensor، مثل نص [ResNet](https://github.com/pytorch/xla/blob/1dc78948c0c9d018d8d0d2b4cce912552ab27083/test/spmd/test_train_spmd_imagenet.py) النصي هذا. ولتنفيذ النص البرمجي باستخدام SPMD، نعتمد على "torchrun":

```
PJRT_DEVICE=CUDA \
torchrun \
--nnodes=${NUM_GPU_MACHINES} \
--node_rank=${RANK_OF_CURRENT_MACHINE} \
--nproc_per_node=1 \
--rdzv_endpoint="<MACHINE_0_IP_ADDRESS>:<PORT>" \
training_or_inference_script_using_spmd.py
```

- `--nnodes`: عدد آلات GPU التي سيتم استخدامها.
- `--node_rank`: فهرس آلات GPU الحالية. ويمكن أن تكون القيمة 0 أو 1 أو ... أو ${NUMBER_GPU_VM}-1.
- `--nproc_per_node`: يجب أن تكون القيمة 1 بسبب متطلبات SPMD.
- `--rdzv_endpoint`: نقطة نهاية آلة GPU ذات node_rank==0، على شكل `host:port`. وسيكون المضيف هو عنوان بروتوكول الإنترنت الداخلي. ويمكن أن يكون "port" أي منفذ متاح على الآلة. وبالنسبة للتدريب/الاستدلال للعقدة الواحدة، يمكن إغفال هذا المعامل.

على سبيل المثال، إذا كنت ترغب في تدريب نموذج ResNet على آلتين GPU باستخدام SPMD، فيمكنك تشغيل النص البرمجي أدناه على الآلة الأولى:

```
XLA_USE_SPMD=1 PJRT_DEVICE=CUDA \
torchrun \
--nnodes=2 \
--node_rank=0 \
--nproc_per_node=1 \
--rdzv_endpoint="<MACHINE_0_INTERNAL_IP_ADDRESS>:12355" \
pytorch/xla/test/spmd/test_train_spmd_imagenet.py --fake_data --batch_size 128
```

وقم بتشغيل ما يلي على الآلة الثانية:

```
XLA_USE_SPMD=1 PJRT_DEVICE=CUDA \
torchrun \
--nnodes=2 \
--node_rank=1 \
--nproc_per_node=1 \
--rdzv_endpoint="<MACHINE_0_INTERNAL_IP_ADDRESS>:12355" \
pytorch/xla/test/spmd/test_train_spmd_imagenet.py --fake_data --batch_size 128
```

لمزيد من المعلومات، يرجى الرجوع إلى [SPMD support on GPU RFC](https://github.com/pytorch/xla/issues/6256).