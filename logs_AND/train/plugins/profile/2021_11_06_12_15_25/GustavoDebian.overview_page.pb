?	?A`??"???A`??"??!?A`??"??	Cs?qH?@Cs?qH?@!Cs?qH?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?A`??"??lC?8??AS?!?uq??Y??jH?c??*	?G?z_?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???-s??!??w? 
U@)??Z?a/??1n$c??T@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?=??j???!??K?5?'@)E?<?\??1??"?a"@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatB?v????!??HՉ@)??sCS??1?n6@:Preprocessing2F
Iterator::Model??gs??!d9?????)???_???1)Y*S????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?IbI????!~z?g????)-??VЄ?1?l,????:Preprocessing2U
Iterator::Model::ParallelMapV2R?r?~?!x?՝"_??)R?r?~?1x?՝"_??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch؃I??	y?!\?݋??)؃I??	y?1\?݋??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?,??Vp?!?6?5???)?,??Vp?1?6?5???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????!-N/w?U@)???VC?n?1?\?7x???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????e?!:@??	???)????e?1:@??	???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::TensorSlice???=?Z?!5???*??)???=?Z?15???*??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::TensorSliceC?8
A?!?T??FS??)C?8
A?1?T??FS??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t18.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9Bs?qH?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	lC?8??lC?8??!lC?8??      ??!       "      ??!       *      ??!       2	S?!?uq??S?!?uq??!S?!?uq??:      ??!       B      ??!       J	??jH?c????jH?c??!??jH?c??R      ??!       Z	??jH?c????jH?c??!??jH?c??JCPU_ONLYYBs?qH?@b Y      Y@q????=@"?
both?Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t18.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQ2"CPU: B 