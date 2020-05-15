# Prepare programming environment

## NVIDIA Driver

Check the GPU status on system

```bash
$ nvidia-smi
Thu Sep  5 09:52:45 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.130                Driver Version: 384.130                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1050    Off  | 00000000:01:00.0 Off |                  N/A |
| 20%   35C    P8   ERR! /  75W |    596MiB /  1999MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1106      G   /usr/lib/xorg/Xorg                           236MiB |
|    0      1784      G   compiz                                       160MiB |
|    0      2491      G   ...quest-channel-token=3235289642966474067   115MiB |
|    0      4362      G   ...quest-channel-token=8255510023286970025    82MiB |
+-----------------------------------------------------------------------------+
```


## Conda environment of Python 3.6

```bash
$ conda create -n ts python=3.6
```

## Install Tensorflow GPU

We using **tensorflow-gpu 1.12** for NVIDIA GTX1050/GTX1050Ti on demo Ubuntu 16.4 desktop.

**Tensorflow-gpu 1.12** must be correspond with **CUDA 9** and **CUDNN 7** as following command. **H5PY** is the file system model of **Keras**.

```bash
$ conda install \
tensorflow-gpu==1.12 \
cudatoolkit==9.0 \
cudnn=7.1.2 \
h5py
```

If success, following output will be delivered when using Python at terminal command line.

```
$ python
Python 3.6.9 |Anaconda, Inc.| (default, Jul 30 2019, 19:07:31) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.Session()
2019-09-05 09:40:30.154172: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-09-05 09:40:30.261480: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-09-05 09:40:30.261891: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.455
pciBusID: 0000:01:00.0
totalMemory: 1.95GiB freeMemory: 1.35GiB
2019-09-05 09:40:30.261910: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-09-05 09:40:30.487837: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-09-05 09:40:30.487870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-09-05 09:40:30.487877: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-09-05 09:40:30.487986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1093 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
<tensorflow.python.client.session.Session object at 0x7f59008e0278>
>>> 
```

We must be received a object of tensorflow session as:

```
<tensorflow.python.client.session.Session object at 0x7f59008e0278>
```

After success to get Tensorflow-GPU, we can using Keras version **2.2.4**.

```bash
$ conda install -c anaconda keras
```

After all, we can ```conda install ...``` any of enable conda package.

# Workflow API

# List all models

```
API:
    GET /api/v1/models

Return: list of ID of models
```

## Create model

``` 
API:
    POST /api/v1/models
        {
            "sensor": "sensor-xxx",
            "algorithm_model": "ae-lstm",
            "ts_field": "IN_BYTES"
        }

Return:
    {
        "model_id": sensor-xxx_ae-lstm_in-bytes,
        "model_path": '/home/cuongbv/Project/tial-ml/soc_tsa/models/sensor-xxx_ae-lstm_in-bytes.h5',
    }

Description:
    sensor: sensor-xxx
    algorithm_model:    ae-lstm (autoencoder + lstm)
                        vae-lstm (variational autoencoder + lstm)
                        stat (statistical)
    ts_field: IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS
    Nếu model đã tồn tại, không tạo model mới đè lên mà trả về luôn 
```

## Training model

```
API:
    PUT api/v1/models/
    {
        "model_id": "sensor-xxx_ae-lstm_in-bytes",
    }

Return: Đang tìm cách return cái mse_loss trên console mà đéo biết cách lấy ra
    response = {
        "model_id": "sensor-xxx_ae-lstm_in-bytes",
        "model_path": '/home/cuongbv/Project/tial-ml/soc_tsa/models/sensor-xxx_ae-lstm_in-bytes.h5',
        "mse_loss": "loss of model after training"
    }

Description:
    Kiểm tra "model_id" có tồn tại hay không, và huấn luyện nếu tồn tại.
    Kéo ữ liệu 7 ngày gần nhất về để huấn luyện 
```

## Forecast 

```
API:
    GET api/v1/models/forecast/?model_id=sensor-xxx_ae-lstm_in-bytes&forecast_hours=1

Return:
    1 list (hoặc numpy array) time series data

Description:
    Chạy forecast cho model_id tương ứng với thời gian dự đoán forecast_hours cụ thể.
```

## Forecast the past

```
API:
    GET api/v1/models/forecast_past/?model_id=sensor-xxx_ae-lstm_in-bytes&forecast_hours=1

Return:
    1 list (hoặc numpy array) time series data

Description:
    Chạy forecast cho model_id tương ứng với thời gian dự đoán forecast_hours cụ thể.
```


# Coding note

1. In code, using function ```clear_session()``` on ```from keras.backend import clear_session``` to clear GPU session after per time running API. This will avoid ```out of memory``` error of GPU.

# Scaling

## Add a new database input

- In ```from soc_tsa.data_input.data_input```, the **Strategy Pattern** is used for the method that get input data. New database is corresponding with a new function.
- ```def elasticsearch_input(self, sensor, algorithm_model, ts_field)``` is Elasticsearch input.

## Add a new handler for input data

- In ```from soc_tsa.factory```, the **Factory Pattern** is used for create a object which will handle the input data.
- In ```from soc_tsa.data_output```, the original class is in **data_output.py**. Any follow class will inheritance from the original class.
- file **dlearn_model.py** is corresponsding with Deep Learning model (Autoencoder + LSTM)
- file **stat_model.py** is corresponsding with Statistical model (ARIMA + LSTM)
- New handling style data will corressponsding with new Python class. Add the new option into **the factory** in ```from soc_tsa.factory.data_output_factory```.

## Add a new Machine Learning model

- In ```from soc_tsa.factory```, the **Factory Pattern** is used for create a object which will create corresponsding model to training/forecasing with input data.
- In ``from soc_tsa.model```, each folder is corresponding with 1 maching learning model.
- ```from soc_tsa.model.autoencoder.stacked_lstm``` is corresponding with the Autoencoder + LSTM model.
- ```from so_tsa.model.statistical.sax_lstm``` is corresponding with the ARIMA + LSTM model.
- 2 above module is **Model Generator**, new Model Generator will corresponding with new folder and new Python class of that Model Generator.
- Add the new option into **the factory** in ```from soc_tsa.factory.model_factory```.

