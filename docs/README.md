## نشر وثائق لإصدار جديد

لم يتم تشغيل مهمة CI "pytorch_xla_linux_debian11_and_push_doc" المحددة للتشغيل على فروع "release/*" على فروع الإصدار بسبب إعداد "Build pull requests فقط". سيؤدي إيقاف تشغيل "Build pull requests فقط" إلى زيادة كبيرة في حجم الوظائف، وهو أمر غير ضروري في كثير من الأحيان. نحن في انتظار [طلب الميزة](https://ideas.circleci.com/ideas/CCI-I-215) هذا ليتم تنفيذه حتى نتمكن من تجاوز هذا الإعداد على بعض الفروع.

قبل توفر الميزة على جانب CircleCi، سنستخدم عملية يدوية لنشر الوثائق الخاصة بالإصدار. لا تزال [وثائق فرع الماستر](http://pytorch.org/xla/master/) يتم تحديثها تلقائيًا بواسطة مهمة CI. ولكن سيتعين علينا الالتزام يدويًا بالوثائق المُصدَّرة الجديدة والإشارة إلى http://pytorch.org/xla إلى وثائق الإصدار المستقر الجديد.

خذ الإصدار 2.3 كمثال:

```
# قم ببناء pytorch/pytorch:release/2.3 و pytorch/xla:release/2.3 على التوالي.
# في pytorch/xla/docs
./docs_build.sh
git clone -b gh-pages https://github.com/pytorch/xla.git /tmp/xla
cp -r build/* /tmp/xla/release/2.3
cd /tmp/xla
# تحديث `redirect_url` في index.md
git add .
git commit -m "نشر وثائق 2.3."
git push origin gh-pages
```

## إضافة وثائق جديدة

لإضافة وثيقة جديدة، يرجى إنشاء ملف `.md` في هذا الدليل. لجعل هذه الوثيقة تظهر في [صفحات الوثائق](https://pytorch.org/xla/master/index.html)، يرجى إضافة ملف `.rst` في `source/`.

## إضافة صور في المستند

يرجى إضافة صورك إلى كل من `_static/img/` و `source/_static/img/` حتى تظهر الصور بشكل صحيح في ملفات Markdown وكذلك في [صفحات الوثائق](https://pytorch.org/xla/master/index.html).