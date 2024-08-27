# المكونات الإضافية للأجهزة المخصصة

يدعم PyTorch/XLA الأجهزة المخصصة من خلال واجهة برمجة تطبيقات C الخاصة بـ PJRT في OpenXLA. ويدعم فريق PyTorch/XLA مباشرة المكونات الإضافية لجهاز Cloud TPU (`libtpu`) و GPU ([OpenXLA](https://github.com/openxla/xla/tree/main/xla/pjrt/gpu)). وقد تستخدم JAX وTF نفس المكونات الإضافية أيضًا.

## تنفيذ مكون PJRT إضافي

قد تكون المكونات الإضافية لواجهة برمجة تطبيقات C في PJRT مفتوحة المصدر أو مغلقة المصدر. وهي تحتوي على جزأين:

1. ثنائي يعرض تنفيذ واجهة برمجة تطبيقات C لـ PJRT. يمكن مشاركة هذا الجزء مع JAX و TensorFlow.
2. حزمة Python تحتوي على الثنائي المذكور أعلاه، بالإضافة إلى تنفيذ لـ `DevicePlugin` واجهة برمجة تطبيقات Python الخاصة بنا، والتي تتعامل مع الإعداد الإضافي.

### تنفيذ واجهة برمجة تطبيقات C لـ PJRT

باختصار، يجب عليك تنفيذ [`PjRtClient`](<https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_client.h>) يحتوي على مترجم XLA ووقت تشغيل لجهازك. يتم عكس واجهة PJRT ++ في C في [`PJRT_Api`](<https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h>).

الخيار الأكثر بساطة هو تنفيذ المكون الإضافي الخاص بك في C ++ و [لفه](<https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_wrapper_impl.h>) كتطبيق واجهة برمجة تطبيقات C. تشرح هذه العملية بالتفصيل في [وثائق OpenXLA](<https://openxla.org/xla/pjrt_integration#how_to_integrate_with_pjrt>).

للحصول على مثال ملموس، راجع مثال [مكون CPU الإضافي](../plugins/cpu). ([تنفيذ OpenXLA](<https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_cpu_internal.cc>)).

### حزمة PyTorch/XLA المكون الإضافي

في هذه المرحلة، يجب أن يكون لديك ثنائي مكون إضافي PJRT وظيفي، والذي يمكنك اختباره باستخدام نوع جهاز `LIBRARY` الوهمي. على سبيل المثال:

```
$ PJRT_DEVICE=LIBRARY PJRT_LIBRARY_PATH=/path/to/your/plugin.so python
>>> import torch_xla
>>> torch_xla.devices()
# افتراض وجود 4 أجهزة. قد تختلف أجهزة الأجهزة الخاصة بك.
[device(type='xla', index=0), device(type='xla', index=1), device(type='xla', index=2), device(type='xla', index=3)]
```

لتسجيل نوع جهازك تلقائيًا للمستخدمين وللتعامل مع الإعداد الإضافي للعديد من العمليات، على سبيل المثال، يمكنك تنفيذ واجهة برمجة تطبيقات Python لـ `DevicePlugin`. تحتوي حزم المكونات الإضافية لـ PyTorch/XLA على مكونين رئيسيين:

1. تنفيذ `DevicePlugin` الذي يوفر (على الأقل) مسار الثنائي المكون الإضافي الخاص بك. على سبيل المثال:

```
class CpuPlugin(plugins.DevicePlugin):
    def library_path(self) -> str:
        return os.path.join(
            os.path.dirname(__file__), 'lib', 'pjrt_c_api_cpu_plugin.so'
        )
```

2. `torch_xla.plugins` [نقطة دخول](<https://setuptools.pypa.io/en/latest/userguide/entry_point.html>) تحدد `DevicePlugin` الخاص بك. على سبيل المثال، لتسجيل نوع الجهاز `EXAMPLE` في `pyproject.toml`:

```
[project.entry-points."torch_xla.plugins"]
example = "torch_xla_cpu_plugin:CpuPlugin"
```

مع تثبيت الحزمة الخاصة بك، يمكنك بعد ذلك استخدام جهاز `EXAMPLE` الخاص بك مباشرة:

```
$ PJRT_DEVICE=EXAMPLE python
>>> import torch_xla
>>> torch_xla.devices()
[device(type='xla', index=0), device(type='xla', index=1), device(type='xla', index=2), device(type='xla', index=3)]
```

يوفر [`DevicePlugin`](<https://github.com/pytorch/xla/blob/master/torch_xla/experimental/plugins.py>) نقاط تمديد إضافية لتهيئة العمليات المتعددة وخيارات العميل. تعد واجهة برمجة التطبيقات حاليًا في حالة تجريبية، ولكن من المتوقع أن تصبح مستقرة في إصدار مستقبلي.