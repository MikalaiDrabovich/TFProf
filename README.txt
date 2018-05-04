TFProf

An example of how to use the default profiler in TensorFlow, provide options (parameters, time) 
and generate automatic 'advice, though the results may or may not not be actionable. 
In the latter case you may want to take a look at 
https://github.com/MikalaiDrabovich/TensorScope


Output of TFProf for training AlexNet on 1060

=========================Options=============================
-max_depth             '     10000
-min_bytes                  1
-min_peak_bytes             0
-min_residual_bytes         0
-min_output_bytes           0
-min_micros                 1
-min_accelerator_micros     0
-min_cpu_micros             0
-min_params                 0
-min_float_ops              0
-min_occurrence             0
-step                     
-order_by                   micros
-account_type_regexes       .*
-start_name_regexes         .*
-trim_name_regexes          
-show_name_regexes          .*
-hide_name_regexes          
-account_displayed_op_only  true
-select                     bytes,micros
-output                     stdout:

==================Model Analysis Report======================

Doc:
op: The nodes are operation kernel type, such as MatMul, Conv2D. Graph nodes belonging to the same type are aggregated together.
requested bytes: The memory requested by the operation, accumulatively.
total execution time: Sum of accelerator execution time and cpu execution time.
cpu execution time: The time from the start to the end of the operation. It's the sum of actual cpu run time plus the time that it spends waiting if part of computation is launched asynchronously.
accelerator execution time: Time spent executing on the accelerator. This is normally measured by the actual hardware library.

Profile:
node name | requested bytes | total execution time | accelerator execution time | cpu execution time
Conv2D                      3243.07MB (100.00%, 29.36%),     138.63ms (100.00%, 32.18%),      56.67ms (100.00%, 19.63%),      81.96ms (100.00%, 57.65%)
Conv2DBackpropFilter         2951.66MB (70.64%, 26.72%),       90.85ms (67.82%, 21.09%),       61.01ms (80.37%, 21.14%),       29.84ms (42.35%, 20.99%)
Conv2DBackpropInput          2919.67MB (43.92%, 26.43%),       73.72ms (46.73%, 17.11%),       44.12ms (59.23%, 15.29%),       29.61ms (21.36%, 20.82%)
LRN                            226.49MB (17.49%, 2.05%),       44.55ms (29.62%, 10.34%),       44.52ms (43.94%, 15.43%),            32us (0.53%, 0.02%)
LRNGrad                        174.42MB (15.44%, 1.58%),       43.49ms (19.28%, 10.09%),       43.45ms (28.51%, 15.06%),            36us (0.51%, 0.03%)
MaxPoolGrad                    219.11MB (13.86%, 1.98%),          5.17ms (9.18%, 1.20%),         5.10ms (13.46%, 1.77%),            73us (0.49%, 0.05%)
ReluGrad                              0B (0.00%, 0.00%),          4.97ms (7.98%, 1.15%),         4.92ms (11.69%, 1.70%),            57us (0.44%, 0.04%)
BiasAdd                               0B (0.00%, 0.00%),          4.22ms (6.83%, 0.98%),          4.16ms (9.99%, 1.44%),            53us (0.40%, 0.04%)
Relu                                  0B (0.00%, 0.00%),          3.33ms (5.85%, 0.77%),          3.29ms (8.55%, 1.14%),            43us (0.36%, 0.03%)
BiasAddGrad                      4.61KB (11.87%, 0.00%),          3.26ms (5.07%, 0.76%),          3.17ms (7.41%, 1.10%),            83us (0.33%, 0.06%)
MaxPool                         45.22MB (11.87%, 0.41%),          2.53ms (4.32%, 0.59%),          2.47ms (6.31%, 0.86%),            60us (0.27%, 0.04%)
conv1/Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer       77.07MB (11.46%, 0.70%),          1.92ms (3.73%, 0.45%),          1.86ms (5.45%, 0.64%),            60us (0.23%, 0.04%)
gradients/conv1_grad/ReluGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer      102.76MB (10.77%, 0.93%),          1.39ms (3.28%, 0.32%),          1.38ms (4.81%, 0.48%),            11us (0.18%, 0.01%)
gradients/conv1_grad/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer       102.76MB (9.84%, 0.93%),          1.38ms (2.96%, 0.32%),          1.36ms (4.33%, 0.47%),            13us (0.18%, 0.01%)
pool1-0-TransposeNHWCToNCHW-LayoutOptimizer       102.76MB (8.91%, 0.93%),          1.37ms (2.64%, 0.32%),          1.36ms (3.86%, 0.47%),            11us (0.17%, 0.01%)
conv1-0-0-TransposeNCHWToNHWC-LayoutOptimizer       102.76MB (7.98%, 0.93%),          1.37ms (2.32%, 0.32%),          1.35ms (3.39%, 0.47%),            13us (0.16%, 0.01%)
gradients/pool1_grad/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer       102.76MB (7.05%, 0.93%),          1.36ms (2.01%, 0.32%),          1.35ms (2.92%, 0.47%),            11us (0.15%, 0.01%)
gradients/conv2_grad/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer        71.66MB (6.12%, 0.65%),           968us (1.69%, 0.22%),           954us (2.45%, 0.33%),            14us (0.14%, 0.01%)
gradients/conv2_grad/ReluGrad-0-2-TransposeNCHWToNHWC-LayoutOptimizer        71.66MB (5.47%, 0.65%),           962us (1.47%, 0.22%),           947us (2.12%, 0.33%),            14us (0.13%, 0.01%)
pool2-0-TransposeNHWCToNCHW-LayoutOptimizer        71.66MB (4.82%, 0.65%),           961us (1.24%, 0.22%),           950us (1.79%, 0.33%),            10us (0.12%, 0.01%)
gradients/pool2_grad/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer       101.87MB (4.17%, 0.92%),           957us (1.02%, 0.22%),           944us (1.46%, 0.33%),            12us (0.12%, 0.01%)
conv2-1-1-TransposeNCHWToNHWC-LayoutOptimizer        71.66MB (3.25%, 0.65%),           954us (0.80%, 0.22%),           945us (1.13%, 0.33%),             9us (0.11%, 0.01%)
conv3-0-0-TransposeNCHWToNHWC-LayoutOptimizer        33.23MB (2.60%, 0.30%),           455us (0.58%, 0.11%),           440us (0.81%, 0.15%),            14us (0.10%, 0.01%)
gradients/conv3_grad/ReluGrad-0-2-TransposeNCHWToNHWC-LayoutOptimizer        33.23MB (2.30%, 0.30%),           454us (0.47%, 0.11%),           441us (0.65%, 0.15%),            13us (0.09%, 0.01%)
gradients/conv5_grad/ReluGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer        22.15MB (2.00%, 0.20%),           336us (0.36%, 0.08%),           323us (0.50%, 0.11%),            12us (0.08%, 0.01%)
pool1-1-0-TransposeNCHWToNHWC-LayoutOptimizer        45.76MB (1.80%, 0.41%),           333us (0.29%, 0.08%),           317us (0.39%, 0.11%),            15us (0.07%, 0.01%)
gradients/conv4_grad/ReluGrad-0-2-TransposeNCHWToNHWC-LayoutOptimizer        22.15MB (1.38%, 0.20%),           306us (0.21%, 0.07%),           293us (0.28%, 0.10%),            13us (0.06%, 0.01%)
conv4-1-0-TransposeNCHWToNHWC-LayoutOptimizer        22.15MB (1.18%, 0.20%),           306us (0.14%, 0.07%),           294us (0.18%, 0.10%),            12us (0.05%, 0.01%)
pool2-1-0-TransposeNCHWToNHWC-LayoutOptimizer        16.61MB (0.98%, 0.15%),           235us (0.07%, 0.05%),           222us (0.08%, 0.08%),            12us (0.05%, 0.01%)
VariableV2                       86.95MB (0.83%, 0.79%),            34us (0.01%, 0.01%),             0us (0.00%, 0.00%),            34us (0.04%, 0.02%)
Mul                               4.72MB (0.04%, 0.04%),            20us (0.00%, 0.00%),             0us (0.00%, 0.00%),            20us (0.01%, 0.01%)

======================End of Report==========================


generating trace file. /job:localhost/replica:0/task:0/device:gpu:0 peak memory: 1116.80 MB

Timeline file is written to timeline_tf_profiler.json_100. Open a Chrome browser, enter URL chrome://tracing and load the timeline file.

2018-05-03 21:19:52.983108: step 90, duration = 0.274 2018-05-03 21:19:54.921238: Forward-backward across 100 steps, 0.221 +/- 0.018 sec / batch

ExpensiveOperationChecker: top 1 operation type: Conv2D, cpu: 81.96ms, accelerator: 56.67ms, total: 138.63ms (32.18%) top 2 operation type: Conv2DBackpropFilter, cpu: 29.84ms, accelerator: 61.01ms, total: 90.85ms (21.09%) top 3 operation type: Conv2DBackpropInput, cpu: 29.61ms, accelerator: 44.12ms, total: 73.72ms (17.11%) top 1 graph node: gradients, cpu: 0us, accelerator: 0us, total: 0us top 2 graph node: conv1, cpu: 11us, accelerator: 1.34ms, total: 1.35ms top 3 graph node: conv2, cpu: 8us, accelerator: 934us, total: 943us alexnet_benchmark_orig.py:313: (gradient), cpu: 59.73ms, accelerator: 161.76ms, total: 221.49ms alexnet_benchmark_orig.py:313:, cpu: 82.20ms, accelerator: 111.11ms, total: 193.32ms

OperationChecker: Found operation using NHWC data_format on GPU. Maybe NCHW is faster.

AcceleratorUtilizationChecker: device: /job:localhost/replica:0/task:0/device:gpu:0 utilization: 1.00


