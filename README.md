# Quantization on MXNet       
Quantization is one of popular compression algorithms in deep learning now. More and more hardwares and softwares 
support quantization, but as we know, it is troublesome that they usually adopt different strategies to quantize.    
    
Here is a tool to help developers simulate quantization with various strategies(signed or unsigned, bits width, 
one-side distribution or not, etc). What's more, quantization aware train is also provided, which will help you recover 
performance of quantized models, especially for efficient ones like MobileNet.

## Simulate quantization     
A tool is provided to simulate quantization for CNN models.     

### Usage
For example, simulate quantization for mobilnet1.0,          
```bash
python simulate_quantization.py --model=mobilnet1.0
```
* Only **per-layer** quantization is supported yet.
* Only **min-max linear** range is supported yet.         
* You can specify **bit-width**, **sign or not**, **one-side-distribution or not**  for input-quantization and 
weight-quantization.
* Quantize input **online** and **offline** are both supported.
* Only calibrate via update EMA for input_min and input_max on subset of trainset for input offline-quantization.
* All pretrained models are provided by gluon-cv.
* More usages see the help message. 
    ```bash
    (base) yahei@Server2:~/tmp/RT_Quantization.MXNet$ python simulate_quantization.py -h
    usage: simulate_quantization.py [-h] [--model MODEL] [--print-model]
                                    [--list-models] [--use-gpu USE_GPU]
                                    [--dataset {imagenet,cifar10}] [--use-gn]
                                    [--batch-norm] [--use-se] [--last-gamma]
                                    [--fake-bn] [--weight-dtype {int,uint}]
                                    [--weight-bits-width WEIGHT_BITS_WIDTH]
                                    [--weight-one-side {false,true}]
                                    [--input-dtype {float,int,uint}]
                                    [--input-bits-width INPUT_BITS_WIDTH]
                                    [--input-one-side {false,true}]
                                    [-j NUM_WORKERS] [--batch-size BATCH_SIZE]
                                    [--num-sample NUM_SAMPLE]
                                    [--quantize-input-offline]
                                    [--calib-epoch CALIB_EPOCH] [--show-warning]
                                    [--disable-cudnn-autotune] [--eval-per-calib]
                                    [--exclude-first-conv {false,true}]
    
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
      --weight-dtype {int,uint}
                            data type to simulate for weights (default: uint)
      --weight-bits-width WEIGHT_BITS_WIDTH
                            bits width of weight to quantize into.
      --weight-one-side {false,true}
                            quantize weights as one-side uint or not. (default:
                            false)
      --input-dtype {float,int,uint}
                            data type to simulate for inputs. (default: uint)
      --input-bits-width INPUT_BITS_WIDTH
                            bits width of input to quantize into.
      --input-one-side {false,true}
                            quantize inputs as one-side uint or not. (default:
                            true)
      -j NUM_WORKERS, --num-data-workers NUM_WORKERS
                            number of preprocessing workers (default: 4)
      --batch-size BATCH_SIZE
                            evaluate batch size per device (CPU/GPU). (default:
                            128)
      --num-sample NUM_SAMPLE
                            number of samples for every class in trainset.
                            (default: 10)
      --quantize-input-offline
                            calibrate via EMA on trainset and quantize input
                            offline.
      --calib-epoch CALIB_EPOCH
                            number of epoches to calibrate via EMA on trainset.
                            (default: 3)
      --show-warning        show warning messages.
      --disable-cudnn-autotune
                            disable mxnet cudnn autotune to find the best
                            convolution algorithm.
      --eval-per-calib      evaluate once after every calibration.
      --exclude-first-conv {false,true}
                            exclude first convolution layer when quantize.
                            (default: true)
    ```    

### Results      
| IN dtype | IN one-side | IN online/OFFLINE | W dtype | W one-side | exclude 1st conv | M-Top1 Acc | R-Top1 Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|float32|/|/|float32|/|/|73.28%|74.37%|
|uint8|✔|online|uint8|✖|✔|72.00%|74.31%|
|uint8|✔|online|uint8|✖|✖|46.92%|52.87%|
|uint8|✔|online|int8|/|✔|70.96%|74.22%|
|uint8|✔|online|int8|/|✖|44.44%|58.67%|
|uint8|✔|OFFLINE|uint8|✖|✔|72.01%|74.22%|
|uint8|✔|OFFLINE|int8|/|✔|70.82%|74.26%|

* **IN**: INput, **W**: Weight
* **M-Top1 Acc**: Top-1 Acc of MobileNetv1-1.0, **R-Top1 Acc**: Top-1 Acc of ResNet34-v1
* Inputs is usually quantized into unsigned inter one-side distribution since outputs of ReLU >= 0.
* Quantize weights into one-side distribution is not appropriate.
* When quantize inputs offline, the range of input is calibrated thrice on subset of trainset, which contains 10000 
images(10 per class).

## Quantize Aware Train
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
                        # such as the first conv and the last fc
    convert_model(net, exclude)
    # convert_model(net, exclude, convert_fn)   # if need to specify converter
    ```
    By default,     
    1. Convert **Conv2D**
        1. Quantize inputs into uint8 with one-side distribution.
        2. Quantize weights with simple strategy of max-min into int8.
    2. Do nothing for **BatchNorm** and **Activiation(ReLU)**.
3. Initialize all quantized parameters.       
    ```python
    from quantize.initialize import qparams_init as qinit
    qinit(net)
    ```
4. Train as usual.
    Note that you should update ema data after forward.      
    ```python
    with autograd.record():
        outputs = net(X)
        loss = loss_func(outputs, y)
    net.update_ema()   # update ema for input and fake batchnorm
    ```
    What's more, you can also switch quantize online or offline as follow:     
    ```python
    net.quantize_input_online()
    net.quantize_input_offline()
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
I've tested mobilenet_v1_1.0 and resnet50_v1 with `Adam` optimizer and no augments on CIFAR100 dataset.    
        
| Quantization          | MobileNet_1_0_ReLU | MobileNet_1_0_ReLU6 | ResNet50_v1 | 
|        :---:          |           :---:    |         :---:       |    :---:    |
| FP32                  | 83.45%             | 84.20%              | 89.35%      |
| UINT8_ONLINE          | 76.61%             | 77.66%              | 89.11%      |
| UINT8_OFFLINE_CALIB   | 72.10%             | 77.44%              | 88.96%      |
| UINT8_OFFLINE_RETRAIN | 80.72%             | 83.03%              | /           |
| UINT8_OFFLINE_FAKEBN  | 80.52%             | 83.00%              | /           |

***Only per-layer quantization is supported yet.***     
   
**ONLINE**, **OFFLINE**: quantize activation online or offline.    
**CALIB**: calibrate quantized parameters of activation with trainset.    
**RETRAIN**: quantize aware training but without fake batchnrom.     
**FAKEBN**: quantize aware training with fake batchnorm.      
      
***More details refer to 《[MXNet上的重训练量化 | Hey~YaHei!](http://hey-yahei.cn/2019/01/23/MXNet-RT_Quantization/)》.***
