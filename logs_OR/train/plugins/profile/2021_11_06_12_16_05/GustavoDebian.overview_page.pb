?	y????^@y????^@!y????^@	y
???@y
???@!y
???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$y????^@仔?d??Au?????@Y?8?????*	A`??"?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?B????!?g?C?DD@)???x??1?[???AC@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??s?????!?'=f??8@)8??w???1r?j??t8@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??ݓ????!j??? :@)???????1$?4?E2@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?$?)? ??!?????@)?j?ѯ??1?Y? B?@:Preprocessing2F
Iterator::Model?U??6o??!/n?? @)?(??/???1IshB?	@:Preprocessing2U
Iterator::Model::ParallelMapV2??0|D??!*Ҝ??>@)??0|D??1*Ҝ??>@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?SW>????!C?-?T???)?SW>????1C?-?T???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?э?????!q???t???)?э?????1q???t???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Ҥ???!?w{W8}P@)??#*T7w?1J?<E??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?"??]?o?!\˒?g???)?"??]?o?1\˒?g???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::TensorSlice	?^)?`?!Yan??)	?^)?`?1Yan??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::TensorSlice)??qX?!$???<??))??qX?1$???<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 35.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9y
???@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	仔?d??仔?d??!仔?d??      ??!       "      ??!       *      ??!       2	u?????@u?????@!u?????@:      ??!       B      ??!       J	?8??????8?????!?8?????R      ??!       Z	?8??????8?????!?8?????JCPU_ONLYYy
???@b Y      Y@q???#Ą8@"?
both?Your program is POTENTIALLY input-bound because 35.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?24.5186% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 