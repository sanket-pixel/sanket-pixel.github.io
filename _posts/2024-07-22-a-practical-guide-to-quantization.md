---
layout: post
title: A practical guide to Quantization
date: 2024-07-21 10:53:00-0400
description: How to exactly quantize models and still not lose accuracy.
thumbnail : /assets/img/blog/blog_6/math.gif
categories: cuda
tag : [nvidia, tensorrt, deep-learning]
giscus_comments: false
related_posts: true
---

#### In this blog, we delve into the practical side of model optimization, focusing on how to leverage TensorRT for INT8 quantization to drastically improve inference speed. By walking through the process step-by-step, we compare pure PyTorch inference, TensorRT optimization, and finally, INT8 quantization with calibration. The results highlight the incredible potential of these techniques, with an over `10x speedup` in performance. Whether you're aiming to deploy deep learning models in production or simply seeking to enhance your understanding of model optimization, this blog provides valuable insights into achieving faster and more efficient inference.

<br>
<div style="width: 100%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_6/compare.png" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
 A practical guide to Quantization
</div>
</div>

In our previous blog post,  [Quantization explained, like you are five](/blog/2024/quantization-explained-like-youre-five/), we covered the theory and intuition behind quantization in detail. If you haven’t read it yet, we highly recommend doing so to gain a solid understanding of the fundamental concepts. This current post serves as a sequel, diving into the practical aspects of quantization. We will guide you through the process of applying quantization to a neural network, demonstrating how it can significantly speed up inference.

In this blog, we would using the resnet based image classfication model as an example. We would be using TensorRT as well for quantization. This constrains this code only for NVIDIA GPU devices. But these same concepts can be extended to other devices and tools without loss of generality. 
If you need a primer on how resnet based image classifciation can be deployed using TensorRT on nvidia devices, refer to previous blog [Have you met TensorRT?](/blog/2023/introduction-to-tensorrt/) where we saw how to use FP16 based TensorRT engine for inference. In this blog we will take it a step further by quantizing the same model to INT8.  

We will go about this using 3 major steps :

**1. Inference with Pure PyTorch**
- Establish a baseline for latency and accuracy using the original PyTorch model.
- Measure inference time and accuracy on a sample dataset.

**2. Inference with TensorRT Engine**
- Convert the PyTorch model to a TensorRT engine using FP16 precision.
- Measure and compare the latency and accuracy to the PyTorch baseline.

**3. Inference with TensorRT Engine (INT8, With Calibration)**
- Perform INT8 quantization with calibration using a calibration dataset.
- Measure latency and accuracy, highlighting the trade-offs and benefits of quantization.


#### 1. Inference with Pure Pytorch 

To establish a baseline for both accuracy and latency, we will first run inference using the ResNet-50 model in PyTorch. This section will cover the preprocessing, inference, and postprocessing steps. The results here will serve as a comparison for the performance improvements we achieve using TensorRT and INT8 quantization in later steps.

##### a. Preprocessing

The preprocessing step involves preparing the input image so that it is suitable for the ResNet-50 model.

```python
import cv2
import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

def preprocess_image(img_path):
    # Transformations for the input data
    transforms = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Read input image
    input_img = cv2.imread(img_path)
    # Apply transformations
    input_data = transforms(input_img)
    batch_data = torch.unsqueeze(input_data, 0)
    return batch_data
```

##### b. Postprocessing
After inference, the output from the model needs to be processed to extract meaningful predictions.

```python
def postprocess(output_data):
    # Get class names
    with open("data/imagenet-classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # Calculate human-readable values by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # Find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # Print the top classes predicted by the model
    while confidences[indices[0][i]] > 10:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1
# Postprocess the output
postprocess(output)
```
##### c. Inference

This step involves running the preprocessed image through the ResNet-50 model to obtain predictions.

```python
from torchvision import models
import torch.backends.cudnn as cudnn
import time
# Load and prepare the model
cudnn.enabled = False
model = models.resnet50(pretrained=True).cuda()
model.eval()
# Preprocess the input image
input = preprocess_image("data/hotdog.jpg").cuda()
# Perform inference
output = model(input)
# Warm-up the GPU
with torch.no_grad():
    for _ in range(5):
        output = model(input)
# Measure latency
latencies = []
iterations = 10
with torch.no_grad():
    for i in range(iterations):
        start_time = time.time()
        output = model(input)
        torch.cuda.synchronize()  # Ensure that all CUDA operations are finished
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
postprocess(output)
average_latency = sum(latencies) / len(latencies)
print(f"Average Latency: {average_latency * 1000:.2f} ms")
```

##### d. Performance
The output from pure Pytorch inference would look something like this. 

```
class: hotdog, hot dog, red hot , confidence: 85.24353790283203 %, index: 934
Average Latency: 3.94 ms
```

#### 2. TensorRT inference 

In this section, we demonstrate how to convert a trained PyTorch model to an ONNX format, then build a TensorRT engine, and finally perform inference using the TensorRT engine. This process significantly optimizes the inference performance on NVIDIA GPUs.

For more detailed explanations on converting PyTorch models to TensorRT engines, refer to our previous blog post, [Have you met TensorRT?](/blog/2023/introduction-to-tensorrt/). 

##### a. Convert PyTorch Model to ONNX

First, we export the PyTorch model to the ONNX format. This intermediate representation serves as a bridge between different deep learning frameworks and tools like TensorRT.

```python
ONNX_FILE_PATH = 'deploy_tools/resnet50.onnx'
torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=['input'],
                  output_names=['output'], export_params=True)
```
##### b. Build TensorRT Engine

Next, we parse the ONNX model and build the TensorRT engine, which is optimized for the target NVIDIA GPU.

```python
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(TRT_LOGGER)

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

parser = trt.OnnxParser(network, TRT_LOGGER)
success = parser.parse_from_file(ONNX_FILE_PATH)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 MiB
serialized_engine = builder.build_serialized_network(network, config)

with open("deploy_tools/resnet50.engine", "wb") as f:
    f.write(serialized_engine)
```
This code snippet handles the following:

- **Logger**: Initializes the TensorRT logger to monitor errors during the engine creation.
- **Builder and Network**: Creates the TensorRT builder and network definition.
- **ONNX Parser**: Parses the ONNX model into the TensorRT network.
- **Engine Serialization**: Serializes the TensorRT engine and saves it to a file for later use.

##### c. Inference with TensorRT Engine

After building the TensorRT engine, we can load it and run inference on the input data. The process involves transferring data to the GPU, executing the engine, and retrieving the results.

```python
runtime = trt.Runtime(TRT_LOGGER)

with open("deploy_tools/resnet50.engine", "rb") as f:
    serialized_engine = f.read()

engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

# Determine dimensions and create memory buffers for inputs/outputs
h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

# Preprocess input image
host_input = np.array(preprocess_image("data/hotdog.jpg").numpy(), dtype=np.float32, order='C')

# Warm-up the GPU
for _ in range(5):
    cuda.memcpy_htod_async(d_input, host_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

# Measure latency
latencies = []
iterations = 1000

for i in range(iterations):
    start_time = time.time()
    cuda.memcpy_htod_async(d_input, host_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    end_time = time.time()
    latency = end_time - start_time
    latencies.append(latency)

average_latency = sum(latencies) / len(latencies)
print(f"Average Latency: {average_latency * 1000:.2f} ms")

# Postprocess and display results
tensorrt_output = torch.Tensor(h_output).unsqueeze(0)
postprocess(tensorrt_output)
torch.cuda.empty_cache()
```

In this step:

- **Runtime and Engine**: We deserialize the saved TensorRT engine and create an execution context.
- **Memory Allocation**: Allocate memory for inputs and outputs on both host (CPU) and device (GPU).
- **Inference**: Perform inference using the TensorRT engine, measuring the latency over multiple iterations.
- **Postprocessing**: Convert the output back to a PyTorch tensor for easier postprocessing and interpretation of the results.

##### d. Performance
The output from TensorRT inference is as follows ( may vary on different GPUs)
```
Average Latency: 2.73 ms
class: hotdog, hot dog, red hot , confidence: 85.29612731933594 %, index: 934
```
With just basic TensorRT engine conversion, we achieved a `1.44x` speedup. Now lets convert it to a quantized INT8 TensorRT engine and see how much speedup we can achieve.

#### 3. Inference with INT8 Quantization

<br>
<div style="width: 100%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_6/quantization.png" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Steps for Quantization
</div>
</div>

In this section, we’ll dive into the heart of the optimization process: calibration and quantization. This process is essential for converting a model to INT8 precision, which can significantly boost inference speed while maintaining a good level of accuracy.

##### a. Implementing a Custom Calibration Class

Before we can convert our model to INT8, we need to calibrate it. Calibration is the process where we run a representative dataset through the model to collect statistical information about its activations. This data is crucial for accurately mapping the floating-point values to INT8 values. To do this in TensorRT, we implement a custom calibration class that extends from the base trt.IInt8EntropyCalibrator2 class.

Here’s how we set up the calibration process:

```python
class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir_path, batch_size=1):
        super(ImageCalibrator, self).__init__()
        self.batch_size = batch_size
        self.current_index = 0
        self.image_paths = self.load_images_paths(image_dir_path)
        self.num_samples = len(self.image_paths)
        self.device_image = cuda.mem_alloc(1*3*224*224*4)

    def load_images_paths(self, image_dir_path):
        return os.listdir(image_dir_path)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_samples:
            return None
        image_path = self.image_paths[self.current_index]
        image_path = "data/calibration_dataset/" + image_path
        image = preprocess_image(image_path)
        cuda.memcpy_htod(self.device_image, np.array(image.numpy(), dtype=np.float32, order='C'))
        self.current_index += self.batch_size
        return [self.device_image]

    def read_calibration_cache(self):
        try:
            with open('calibration.cache', 'rb') as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        with open('calibration.cache', 'wb') as f:
            f.write(cache)
```

What’s Happening Here?

**Initialization**: The class is initialized with the directory path containing calibration images. These images are loaded into memory as the calibration process iterates through them.

**Image Preprocessing**: Each image from the calibration dataset is preprocessed and then copied to the GPU’s memory for use in calibration. The batch size is set to 1 for simplicity, but this can be adjusted based on your needs.

**Handling the Calibration Cache**: TensorRT can cache the calibration data, so you don’t have to recalibrate the model every time you run this process. The cache is read from and written to the disk, allowing for faster reuse of calibration data in future runs.

When calibrating other models, you’ll need to implement a similar class tailored to the specific preprocessing and data handling requirements of your model.

##### b. Building the TensorRT Engine with INT8 Quantization

Once we’ve set up our calibrator, we’re ready to build the TensorRT engine with INT8 precision. The steps involve parsing the model from an ONNX file and configuring the builder to use the INT8 calibration data.

Here’s how it’s done:

```python
ONNX_FILE_PATH = "deploy_tools/resnet50.onnx"
image_dir = "data/calibration_dataset"
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(TRT_LOGGER)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

parser = trt.OnnxParser(network, TRT_LOGGER)
success = parser.parse_from_file(ONNX_FILE_PATH)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

# Assign the custom calibrator
config.int8_calibrator = ImageCalibrator(image_dir)
config.set_flag(trt.BuilderFlag.INT8)

# Build and serialize the INT8 TensorRT engine
serialized_engine = builder.build_serialized_network(network, config)
with open("deploy_tools/resnet_int8.engine", "wb") as f:
    f.write(serialized_engine)
```

Key Points:

- **ONNX Parser**: The model is parsed from the ONNX format, and the network is created using TensorRT.

- **INT8 Calibration**: The custom calibrator class we defined earlier is assigned to the builder configuration. The INT8 flag is set, instructing TensorRT to use the calibration data for quantizing the model.

- **Engine Serialization**: Once the engine is built with INT8 precision, it is serialized and saved to disk. This engine is now optimized for fast inference using 8-bit integer operations.

For more details on the general process of building TensorRT engines, refer to our previous post, "Have you met TensorRT?".

##### c. Running Inference with the INT8-Optimized Engine

Finally, we can run inference using the INT8-optimized TensorRT engine. Here’s how it’s done:

```python

runtime = trt.Runtime(TRT_LOGGER)
with open("deploy_tools/resnet_int8.engine", "rb") as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

# Allocate memory for inputs and outputs
h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

# Preprocess and load the input image
host_input = np.array(preprocess_image("data/hotdog.jpg").numpy(), dtype=np.float32, order='C')

# Warm-up the GPU
for _ in range(5):
    cuda.memcpy_htod_async(d_input, host_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

# Measure latency for INT8 inference
iterations = 1000
latencies = []

for i in range(iterations):
    start_time = time.time()
    cuda.memcpy_htod_async(d_input, host_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    end_time = time.time()
    latencies.append(end_time - start_time)

# Calculate average latency
average_latency = sum(latencies) / len(latencies)
print(f"Average Latency for INT8 Inference: {average_latency * 1000:.2f} ms")

# Post-process the output
tensorrt_output = torch.Tensor(h_output).unsqueeze(0)
postprocess(tensorrt_output)
```

What’s Happening Here?

- **Engine Deserialization**: The INT8 engine is loaded from disk and deserialized into memory.

- **Memory Management**: We allocate memory for inputs and outputs on both the host and device, setting up everything needed to run inference.

- **Inference Execution**: After warming up the GPU, inference is run multiple times to measure latency. Using INT8 should provide a significant reduction in latency compared to FP32 inference.

- **Output Post-processing**: Finally, the output is converted back to a PyTorch tensor and processed just like we did with the FP32 model.

This step demonstrates the power of quantization. With the INT8 engine, we can achieve faster inference times, which is crucial for deploying models in real-time applications.

##### d. Performance
The output from INT8 quantized TensorRT engine is as follows :

```
Average Latency for INT8 Inference: 0.39 ms
class: hotdog, hot dog, red hot , confidence: 93.55050659179688 %, index: 934
```

As we can see, INT8 quantization achieved a whooping `10.25x speedup` as compared to pure Pytorch inference. All the struggle we went through for calibration and quantization was indeed worth it. ;)

#### Summary

In this blog, we explore the practical application of quantization using TensorRT to significantly speed up inference on a ResNet-based image classification model. We begin with a baseline comparison, demonstrating inference using pure PyTorch, which provides a foundational understanding of the model's performance. Next, we transition to using a TensorRT engine, showcasing the initial speed improvements by optimizing the model for NVIDIA GPUs. Finally, we delve into INT8 quantization, applying calibration techniques to maximize efficiency. The results are striking, with a speedup of over 10x compared to the original PyTorch inference, illustrating the power of quantization in real-world scenarios.