	7m?i?J@7m?i?J@!7m?i?J@	?)?4x$@?)?4x$@!?)?4x$@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$7m?i?J@Q??r????A??????Y?_?????*	?&1??@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapU?wE????!???|B@)8???????1?%?ݮ<@:Preprocessing2F
Iterator::Model?z0)>??!?x?Ep:@)0EH????1w??+?1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??A^&??![?E??0@)??A^&??1[?E??0@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapR}?%???!t?e?y?2@)YNB?!??1J/?0@:Preprocessing2U
Iterator::Model::ParallelMapV2??kЗ޲?!? ???!@)??kЗ޲?1? ???!@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?? Z+??!??N?!@)?? Z+??1??N?!@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?w)u?8??!?R??>?@)?&?????1?4??|@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?э????!?ow1@)>?*??1???????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipػ?ޫV??!P????vK@)@??wԘp?1??,???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::TensorSliceO??唀h?!rȤ????)O??唀h?1rȤ????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??@??c?!??????)??@??c?1??????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::TensorSlice???]/MA?!?6?p#2??)???]/MA?1?6?p#2??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?)?4x$@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Q??r????Q??r????!Q??r????      ??!       "      ??!       *      ??!       2	????????????!??????:      ??!       B      ??!       J	?_??????_?????!?_?????R      ??!       Z	?_??????_?????!?_?????JCPU_ONLYY?)?4x$@b 