# Quantization on MXNet       
Quantization is one of popular compression algorithms in deep learning now. More and more hardwares and softwares 
support quantization, but as we know, it is troublesome that they usually adopt different strategies to quantize.    
    
Here is a tool to help developers simulate quantization with various strategies(signed or unsigned, bits width, 
one-side distribution or not, etc). What's more, quantization aware train is also provided, which will help you recover 
performance of quantized models, especially for compact ones like MobileNet.

* [Simulate quantization](#simulate-quantization)
    * [Usage](#usage)
    * [Results](#results)
* [Quantization Aware Training](#quantization-aware-training)
    * [Usage](#usage-1)
    * [Results](#results-1)
    * [Deploy to third-party platform](#deploy-to-third-party-platform)
        * [Tengine](#tengine)
        * [ncnn](#ncnn)

## Simulate quantization     
A tool is provided to simulate quantization for CNN models.     

### Usage
For example, simulate quantization for mobilnet1.0,          
```bash
cd examples
python simulate_quantization.py --model=mobilnet1.0
```
* **Per-layer**, **per-group**, **per-channel** quantizations are supported now.
* For FullyConnection layer, **per-group** and **per-channel** both mean that weights wil be grouped by units.
* Only **min-max linear** range is supported yet.         
* You can specify **bit-width** for input-quantization and weight-quantization.
* Quantize input **online** and **offline** are both supported.
* Calibrate via update EMA for input_max or KL-divergence on subset of trainset for input offline-quantization.
* All pretrained models are provided by gluon-cv.
* More usages see the help message. 
    ```bash
    (base) yahei@Server3:~/tmp/Quantization.MXNet/examples$ python simulate_quantization.py -h
    usage: simulate_quantization.py [-h] [--model MODEL] [--print-model]
                                    [--list-models] [--use-gpu USE_GPU]
                                    [--dataset {imagenet,cifar10}] [--use-gn]
                                    [--batch-norm] [--use-se] [--last-gamma]
                                    [--fake-bn]
                                    [--weight-bits-width WEIGHT_BITS_WIDTH]
                                    [--input-signed INPUT_SIGNED]
                                    [--input-bits-width INPUT_BITS_WIDTH]
                                    [--quant-type {layer,group,channel}]
                                    [-j NUM_WORKERS] [--batch-size BATCH_SIZE]
                                    [--num-sample NUM_SAMPLE]
                                    [--quantize-input-offline]
                                    [--calib-mode {naive,kl}]
                                    [--calib-epoch CALIB_EPOCH]
                                    [--disable-cudnn-autotune] [--eval-per-calib]
                                    [--exclude-first-conv {false,true}]
                                    [--fixed-random-seed {false,true}]
    
    Simulate for quantization.
    
    optional arguments:
      -h, --help            show this help message and exit
      --model MODEL         type of model to use. see vision_model for options.
                            (required)
      --print-model         print the architecture of model.
      --list-models         list all models supported for --model.
      --use-gpu USE_GPU     run model on gpu. (default: cpu)
      --dataset {imagenet,cifar10}
                            dataset to evaluate (default: imagenet)
      --use-gn              whether to use group norm.
      --batch-norm          enable batch normalization or not in vgg. default is
                            false.
      --use-se              use SE layers or not in resnext. default is false.
      --last-gamma          whether to init gamma of the last BN layer in each
                            bottleneck to 0.
      --fake-bn             use fake batchnorm or not.
      --weight-bits-width WEIGHT_BITS_WIDTH
                            bits width of weight to quantize into.
      --input-signed INPUT_SIGNED
                            quantize inputs into int(true) or uint(fasle).
                            (default: false)
      --input-bits-width INPUT_BITS_WIDTH
                            bits width of input to quantize into.
      --quant-type {layer,group,channel}
                            quantize weights on layer/group/channel. (default:
                            layer)
      -j NUM_WORKERS, --num-data-workers NUM_WORKERS
                            number of preprocessing workers (default: 4)
      --batch-size BATCH_SIZE
                            evaluate batch size per device (CPU/GPU). (default:
                            128)
      --num-sample NUM_SAMPLE
                            number of samples for every class in trainset.
                            (default: 5)
      --quantize-input-offline
                            calibrate via EMA on trainset and quantize input
                            offline.
      --calib-mode {naive,kl}
                            how to calibrate inputs. (default: naive)
      --calib-epoch CALIB_EPOCH
                            number of epoches to calibrate via EMA on trainset.
                            (default: 3)
      --disable-cudnn-autotune
                            disable mxnet cudnn autotune to find the best
                            convolution algorithm.
      --eval-per-calib      evaluate once after every calibration.
      --exclude-first-conv {false,true}
                            exclude first convolution layer when quantize.
                            (default: true)
      --fixed-random-seed {false,true}
                            set random_seed=7 for numpy to provide
                            reproducibility. (default: true)
    ```    

### Results      
| IN dtype | IN offline | WT dtype | WT qtype | Merge BN | w/o 1st conv | M-Top1 Acc | R-Top1 Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| float32 | / | float32 | / |   | / | 73.28% | 77.36% |
| uint8 | x | int8 | layer |   | √ | 70.84% | 76.92% |
| uint8 | x | int8 | layer |   |   | 44.57% | 55.97% |
| uint8 | naive | int8 | layer |   | √ | 70.92% | 76.90% |
| int8 | naive | int8 | layer |   | √ | 70.58% | 76.81% |
| uint8 | KL | int8 | layer |   | √ | 70.72% | 77.00% |
| int8 | KL | int8 | layer |   | √ | 70.66% | 76.71% |
| int8 | x | int8 | layer | √ | √ | 15.21% | 76.62% |
| int8 | naive | int8 | layer | √ | √ | 32.70% | 76.61% |
| int8 | KL | int8 | layer | √ | √ | 14.70% | 76.60% |
| uint8 | x | int8 | channel |   | √ | 72.93% | 77.33% |
| uint8 | x | int8 | channel |   |   | 47.80% | 56.21% |
| uint8 | naive | int8 | channel |   | √ | 72.85% | 77.31% |
| int8 | naive | int8 | channel |   | √ | 72.63% | 77.22% |
| uint8 | KL | int8 | channel |   | √ | 72.68% | 77.35% |
| int8 | KL | int8 | channel |   | √ | 72.68% | 77.08% |
| int8 | x | int8 | channel | √ | √ | 72.75% | 77.11% |
| int8 | naive | int8 | channel | √ | √ | 72.04% | 76.69% |
| int8 | KL | int8 | channel | √ | √ | 72.67% | 77.07% |

* **IN**: INput, **WT**: WeighT
* **M-Top1 Acc**: Top-1 Acc of MobileNetv1-1.0, **R-Top1 Acc**: Top-1 Acc of ResNet50-v1
* Inputs is usually quantized into unsigned int with one-side distribution since outputs of ReLU >= 0.
* When quantize inputs offline, the range of input is calibrated thrice on subset of trainset, which contains 5000 
images(5 per class).    
* Merge BatchNorm before quantization seams terrible for per-layer because some `max(abs(weight))` would be much larger after merge bn.
* Convolutions and FullyConnections are both quantized.
* Without fake_bn, calibrate input_max via EMA and KL-divergence both recover acc well. But with fake_bn, calibrate via KL-divergence seems better than EMA. 

## Quantization Aware Training
Reproduce works in paper [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) with the implement of MXNet.
### Usage    
1. Construct your gluon model. For example,     
    ```python
    from mxnet.gluon.model_zoo.vision import mobilenet1_0
    net = mobilenet1_0(pretrained=True)
    ```        
2. Convert the model to fake-quantized edtion. For example,       
    ```python
    from quantize.convert import convert_model
    exclude = [...]     # the blocks that you don't want to quantize
                        # such as the first conv
    convert_fn = {...}
    
    convert_model(net, exclude)
    # convert_model(net, exclude, convert_fn)   # if need to specify converter
    ```
    By default,     
    1. Convert **Conv2D**
        1. Quantize inputs into uint8 with one-side distribution.
        2. Quantize weights(per-layer) with simple strategy of max-min into int8.
        3. Without fake batchnorm.
    2. Convert **Dense**
        1. Quantize inputs into uint8 with one-side distribution.
        2. Quantize weights(per-layer) with simple strategy of max-min into int8.
    3. Do nothing for **BatchNorm** and **Activiation(ReLU)**.
    4. Note that if you use fake_bn, bypass_bn must be set for BatchNorm layer.
3. Initialize all quantized parameters.       
    ```python
    from quantize.initialize import qparams_init
    qparams_init(net)
    ```
4. Train as usual.
    Note that you should update ema data after forward.      
    ```python
    with autograd.record():
        outputs = net(X)
        loss = loss_func(outputs, y)
    net.update_ema()   # update ema for input and fake batchnorm
    trainer.step(batch_size)
    # trainer.step(batch_size, ignore_stale_grad=True)   # if bypass bn
    ```
    What's more, you can also switch quantize online or offline as follow:     
    ```python
    net.quantize_online()
    net.quantize_offline()
    ```
    
<!--
### Freeze(have not tested)    
To help freeze gluon models to symbol, **FreezeHelper** is provided.     
1. Construct gluon model without initialization. For example,      
    ```python
    net = mobilenet1_0(pretrained=False)
    ```     
2. Create a helper with parameter file.      
    ```python
    from quantize.freeze import FreezeHelper
    helper = FreezeHelper(
       net=net,
       params_filename="/path/to/trained/parameters/file",
       input_shape=(1,3,224,224), # [default] (1,3,224,224)
       tmp_filename="/path/to/tmp/file",   # it will be deleted when delete helper
     )
    ```
3. Show all symbols and find out symbols you want to exclude and parameters you want to quantize offline.    
    ```python
    helper.list_symbols()    # show all symbols
    # helper.list_symbols(suffix="fwd")    # show all forward symbols
    # helper.list_symbols(suffix=("weight", "bias"))    # show all weight and bias symbols
    ```
4. Freeze gluon model.     
    ```python
    import mxnet as mx
    exclude = [...]
    offline = [...]
    qsym, qargs, auxes = helper.freeze(exclude, offline, quantize_input_offline=True)
    mx.model.save_checkpoint("/path/to/save", 0, qsym, qargs, auxes)   # if save the model
    ```
-->   
 
### Results    
I've tested mobilenet_v1_1.0 with `Adam` optimizer and no augments on ImageNet(ILSVRC2012) dataset.    
        
| DataType   | QuantType | Offline | Retrain | FakeBN | Top-1 Acc |
| :---:      | :---:     | :---:   | :---:   | :---:  | :---:     |
| float32    | / | / | / | / | 73.28% |
| int8/uint8 | layer |   |   |   | 70.85% |
| int8/uint8 | layer |   | √ |   | 73.01% |
| int8/uint8 | layer | √ |   |   | 70.90% |
| int8/uint8 | layer | √ | √ |   | 73.04% |
| int8/int8 | channel | √ |   | √ | 71.57% |
| int8/int8 | channel | √ | √ | √ | 72.46% |

* The first convolution layer is excluded when quantize.
* Weights are quantized into int8 while inputs int8/uint8.
* No matter whether quantize inputs offline or not, the accuracy can be recovered well via retrain. 
* Only a subset of trainset which contains 10000 images(10 for per class) is used when retrain the net.   
* Retraining seams not very useful for per-channel quantization but a little benefit for fake bn.
   
### Deploy to third-party platform
#### [Tengine](https://github.com/OAID/Tengine)
Tengine support int8-inference for explorer version, you can just export gluon model to mxnet model(with a `.json` file and a `.param` file) with float32 parameters. Tengine will parse it and use int8-inference online automatically.       
Note that, in Tengine,    
1. Weights are quantized into int8.     
2. Inputs(Activations) are quantized into int8 or uint8(optional).
3. BatchNorm would be fused into Convolution, and then quantize weights.
4. Per-group quantization is used.

#### [ncnn](https://github.com/Tencent/ncnn)
ncnn only support int8-inference for caffe model yet, so you should convert your model to caffemodel with [GluonConverter](https://github.com/hey-yahei/GluonConverter) at first.    
Generate scales table just as `examples/mobilenet_gluon2ncnn.ipynb` does and convert caffemodel to ncnnmodel with `caffe2ncnn` tool which is provided by ncnn.     
Note that, in ncnn,
1. Both weights and inputs(activations) are quantized into int8.
2. BatchNorm should be fused into Convolution before you calculate scales for weights(retrain with fake_bn may help recover accuracy).
3. Per-channel quantization is used. 
   
---------------------        
   
***More details refer to 《[MXNet上的重训练量化 | Hey~YaHei!](http://hey-yahei.cn/2019/01/23/MXNet-RT_Quantization/)》.***
