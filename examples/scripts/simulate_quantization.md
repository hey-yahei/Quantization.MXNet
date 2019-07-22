### Scripts for [simulate_quantization.py](../simulate_quantization.py)

#### cifar_resnet56_v1

##### c10_r56_uint4_int4_layer_merge_naive
```
python simulate_quantization.py --model=cifar_resnet56_v1 --input-bits-width=4 --weight-bits-width=4 --dataset=cifar10 --merge-bn --quantize-input-offline --num-sample=500 --calib-mode=naive
```

##### c10_r56_uint4_int4_layer_merge_kl
```
python simulate_quantization.py --model=cifar_resnet56_v1 --input-bits-width=4 --weight-bits-width=4 --dataset=cifar10 --merge-bn --quantize-input-offline --num-sample=500 --calib-mode=kl
```        

##### c10_r56_uint4_int4_channel_merge_naive
```
python simulate_quantization.py --model=cifar_resnet56_v1 --quant-type=channel --input-bits-width=4 --weight-bits-width=4 --dataset=cifar10 --merge-bn --quantize-input-offline --num-sample=500 --calib-mode=naive
```

##### c10_r56_uint4_int4_channel_merge_kl
```
python simulate_quantization.py --model=cifar_resnet56_v1 --quant-type=channel --input-bits-width=4 --weight-bits-width=4 --dataset=cifar10 --use-gpu=1 --merge-bn --quantize-input-offline --num-sample=500 --calib-mode=kl
```
