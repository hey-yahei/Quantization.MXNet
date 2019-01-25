# Quantization Aware Training
Reproduce works in paper [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) with the implement of MXNet.      

## Usage    
### Train    
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
    1. Convert **Conv2D** to the one with input quantization and fake batchnorm.     
    2. Convert **ReLU** to **ReLU6**.       
    3. Bypass **Batchnorm**.      
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
### Freeze    
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
## Result    
I've tested on mobilenet_v1_1.0 and resnet50_v1 with `Adam` optimizer and no augments.    
        
| Quantization          | MobileNet_1_0_ReLU | MobileNet_1_0_ReLU6 | ResNet50_v1 | 
|        :---:          |           :---:    |         :---:       |    :---:    |
| FP32                  | 83.45%             | 84.20%              | 89.35%      |
| UINT8_ONLINE          | 76.61%             | 77.66%              | 89.11%      |
| UINT8_OFFLINE_CALIB   | 72.10%             | 77.44%              | 88.96%      |
| UINT8_OFFLINE_RETRAIN | 80.72%             | 83.03%              | /           |
| UINT8_OFFLINE_FAKEBN  | 80.52%             | 83.00%              | /           |

***Only per-layer quantization is supported now.***     
   
**ONLINE**, **OFFLINE** means activating online or offline.    
**CALIB** means calibrate quantized parameters of activation with trainset.    
**RETRAIN** means quantize aware training without fake batchnrom.     
**FAKEBN** means quantize aware training with fake batchnorm.     

------------------------     
***More details refer to 《[MXNet上的重训练量化 | Hey~YaHei!](http://hey-yahei.cn/2019/01/23/MXNet-RT_Quantization/)》.***
