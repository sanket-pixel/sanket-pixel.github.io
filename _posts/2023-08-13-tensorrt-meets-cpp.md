---
layout: post
title: TensorRT meets C++
date: 2023-08-08 15:53:00-0400
description: TensorRT inference in C++
thumbnail : /assets/img/blog/blog_3/main.png
categories: edge-ai
tag : [nvidia, tensorrt, deep-learning]
giscus_comments: false
related_posts: true
---

#### Building upon the foundations laid in our previous post, "Have you met TensorRT?," where we embarked on a journey into the world of basic concepts using Python, we now delve into the exciting realm of C++. By seamlessly integrating TensorRT with C++, this blog unlocks the potential for readers to effortlessly transition their PyTorch models into a C++ environment. We present an illustrative example of image classification, utilizing the familiar model from our earlier exploration. As the blog unfolds, the power of this integration becomes evident—readers will learn how to read an input image using OpenCV, copy it to the GPU, perform inference to get the output, and copy the output back to the CPU. This sets a strong foundation for utilizing this pipeline with any standard PyTorch model. This blog empowers readers with the knowledge to bridge the gap between two domains, ultimately enabling them to harness the capabilities of TensorRT in the world of C++.

<br> 
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/blog/blog_3/main.png" title="TensorRT meets C++" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    TensorRT meets C++
</div>


Welcome to our blog series where the worlds of TensorRT and C++ converge to revolutionize the AI landscape. In our previous installment, [Have you met TensorRT?](/blog/2023/introduction-to-tensorrt/), we embarked on an exciting exploration of the fundamental concepts, laying the groundwork for the cutting-edge journey we're about to embark upon. Now, with "TensorRT meets C++",  we usher you into a realm of possibilities where the seamless integration of these technologies has profound implications for AI, particularly in the context of robotics and edge computing.

#### Unveiling the Power of Integration

The significance of this integration cannot be understated. While our prior post introduced you to TensorRT's prowess in Python, this blog takes you a step further. By blending TensorRT with the power of C++, we equip you with the skills to transition your PyTorch models seamlessly into a C++ environment. This transition isn't just about speed—although the enhanced inference speed is undoubtedly thrilling—it's about more. It's about delving into the heart of memory management, understanding the intricacies of processor operations, and acquiring a deeper comprehension of how your models interact with the hardware.

#### The Challenge of Copying: Navigating the GPU Highway

With great power comes great responsibility. The marriage of TensorRT and C++ introduces a pivotal challenge: the accurate transfer of data between the CPU and GPU. As we embark on this journey, we delve into the problem of copying data correctly onto the GPU and navigating the intricacies of memory transfers. We dissect this challenge, peeling back the layers to understand how to harmonize these distinct processing units and deliver seamless data flow.

#### Unveiling a New Frontier: From Vision to Robotics

Our blog isn't just about one aspect of AI—it's about a journey that spans diverse domains. This integration is the key that unlocks possibilities beyond image classification. From lidar-based perception to planning-based decision-making, from text-based sentiment analysis to complex deep reinforcement learning, the door to countless applications opens. All these, in the realm of robotics and edge AI where C++ reigns supreme. As you delve into this blog, you're not merely mastering a technology; you're bridging the chasm between your Jupyter notebooks and the robotics that can wield your models.

#### Source from Github
For those interested in exploring the code and gaining a deeper understanding of the concepts discussed in this blog on TensorRT and image classification, you can find the complete source code in the corresponding GitHub repository. The repository link is [this](https://github.com/sanket-pixel/tensorrt_cpp).

#### Pre-requisites and Installation
**1. Hardware requirements**
- NVIDIA GPU

**2. Software requirements**
- Ubuntu >= 18.04
- Python >= 3.8

**3. Installation Guide**
1. Create conda environment and install required python packages.
```
conda create -n trt python=3.8
conda activate trt
pip install -r requirements.txt
```

2. Install TensorRT 
Install TensorRT:

- Download and install NVIDIA CUDA 11.4 or later following the official instructions: [link](https://developer.nvidia.com/cuda-toolkit-archive)

- Download and extract CuDNN library for your CUDA version (>8.9.0) from: [link](https://developer.nvidia.com/cudnn)

- Download and extract NVIDIA TensorRT library for your CUDA version from: [link](https://developer.nvidia.com/nvidia-tensorrt-8x-download). Minimum required version is 8.5. Follow the Installation Guide for your system and ensure Python's part is installed.

- Add the absolute path to CUDA, TensorRT, and CuDNN libs to the environment variable PATH or LD_LIBRARY_PATH.

- Install PyCUDA:
```
pip install pycuda
```

<br> 



#### The Road Ahead: Sections to Explore

As we journey through "TensorRT meets C++," we'll traverse several vital sections:

1. **Intuition**: Lay the groundwork by understanding the synergy between TensorRT and C++.

2. **Inside Deep Dive**: Embark on an exploration of the inner workings, understanding the harmony of memory and processing.

3. **Latency and Consistency**: Quantify the gains—measure the reduced latency that TensorRT offers.

4. **Conclusion**: Weave together the threads of knowledge, leaving you ready to wield the integration with confidence.

Prepare to witness the fusion of AI and robotics—a fusion that empowers you to take your models beyond the notebook and into the real world, where C++ is the language of choice. Let's embark on this transformative journey together.


## 1. Intuition
In the journey of merging the powerful capabilities of TensorRT with the versatile landscape of C++, we embark on a comprehensive exploration of intuitive concepts that form the bedrock of seamless model inference. This section of the blog delves into four key intuitions, each unraveling a layer of understanding that enriches our grasp of the integration process. From traversing the path from a Torch model to C++ inference to uncovering the mechanics where TensorRT and C++ converge, and from unraveling the intricacies of memory operations to mapping RGB channels for perfect alignment—the intuitions explored herein shed light on the intricate dance between theory and application. These insights, rooted in both practical implementation and theoretical foundations, propel us toward mastering the harmonious symphony that is "TensorRT meets C++."

####  1.1. From Torch Model to C++ Inference

In the realm of AI and machine learning, the journey from a PyTorch model within the cozy confines of a Jupyter notebook to real-world applications demands a bridge—a bridge that harmonizes the worlds of deep learning and robust, high-performance C++ execution. Enter TensorRT, the catalyst that transforms this transition from an ambitious leap to a smooth stride. Let us look at an overview of how we make this transition.
 
 The diagram below captures the essence of this journey, illustrating how the torch model transforms into an optimized TensorRT engine and finds its home in the world of C++.

<div style="width: 100%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_3/pt_cpp.drawio.png" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  Going from Pytorch model to Inference in C++
</div>
</div>

<br>

**1: The Torch-to-ONNX Conversion**

At the heart of this transformation lies the conversion of the PyTorch model into the Open Neural Network Exchange (`ONNX`) format. This universal format serves as the lingua franca between frameworks, unlocking the potential to bridge the gap between PyTorch and TensorRT. The Torch-to-ONNX conversion encapsulates the model's architecture and parameters, setting the stage for seamless integration into the C++ landscape.

**2: The ONNX-to-TensorRT Metamorphosis**

With the ONNX representation in hand, in form of an `.onnx` file, we traverse the second leg of our journey—the ONNX-to-TensorRT transformation. Here, the ONNX model metamorphoses into a high-performance TensorRT engine, optimized to harness the prowess of modern GPUs. TensorRT's meticulous optimization techniques prune the neural network, leveraging the inherent parallelism of GPUs for expedited inference without compromising accuracy. This can happen via the `trtexec` tool provided by TensorRT, via the Python API or through the C++ API. Having already covered the Python API implementation in the previous blog, we shall see the C++ API execution later in this blog.

**3: Unveiling C++ Inference**

And now, with the TensorRT engine prepared, in form of a `.engine` file, we navigate the final stretch of our voyage—the integration of the engine into C++ for inference. Armed with the TensorRT-powered engine, C++ becomes the stage where our models perform with astounding efficiency. Leveraging C++'s capabilities, we create a pipeline to read input data, offload it onto the GPU, execute inference, and retrieve the output—effortlessly spanning the distance between deep learning models and real-world applications.



#### 1.2. The Fusion: Where TensorRT Meets C++

In the intricate choreography between TensorRT and C++, a harmonious integration unfolds, paving the way for seamless model inference. As we transition from a TensorRT engine to the dynamic landscape of C++, a meticulous orchestration of steps ensures a blend of precision, speed, and efficiency that bridges the gap between these two powerful realms.

<div style="width: 100%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_3/memory.png" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  Memory Transfers during inference.
</div>
</div>

<br>

**1. Setting the Stage: Input and Output Shapes**

Our journey begins by laying a foundation of understanding—the dimensions of the engine's input and output tensors. These dimensions become the architectural blueprints for memory allocation and data manipulation. Armed with this knowledge, we traverse the intricate memory pathways that interconnect the CPU and GPU.

**2. Memory Allocation: The CPU-GPU Ballet**

With dimensions in hand, the stage shifts to memory allocation. The CPU takes center stage, employing `malloc()` to reserve space for the input and output tensors (`host_input` and `host_output`). Simultaneously, the GPU claims its spot, using `cudaMalloc()` to allocate memory for the tensors (`device_input` and `device_output`). This synchronization lays the groundwork for the fluid movement of data between the CPU and GPU.

**3. Orchestrating Data Flow: Copy and Serialize**

As the memory symphony unfolds, we shift our focus to the heart of data manipulation. The input—an image in this case—is transformed into a flattened array and stored in (`host_input`) that succinctly encapsulates the image data in a 1D structure. This array, a language understood by both the CPU and GPU, prepares for its leap onto the GPU memory stage.


**4. The Leap to GPU: Copy and Sync**

The synchronization is executed with precision. Our flattened array, now embodied as `host_input`, takes its leap from CPU memory to the GPU's allocated memory (`device_input`). This transition is elegantly facilitated by `cudaMemcpy()`, as the image data makes its home within GPU memory, seamlessly bridging the chasm between CPU and GPU.

**5. The Grand Performance: Execution and Output**

The pinnacle of our journey arrives with the TensorRT engine poised on the stage. Equipped with the memory addresses of the input and output residing within the GPU (`device_input` and `device_output`), the engine's inference function takes the command, orchestrating a high-speed performance where calculations unfurl on the parallel prowess of the GPU. The outcome—a meticulously calculated inference output—is etched onto the GPU's at `device_output`.

**6. The Final Flourish: Retrieving the Output**

As the GPU's performance concludes, it's time to unveil the masterpiece—the inference output `device_output`. Guided by `cudaMemcpy()`, the `device_output` elegantly navigates the CPU-GPU bridge, returning to the CPU at `host_output`. Here, the versatile arms of C++ embrace it, ready to be presented to the world—bridging the divide between a PyTorch model and real-world applications.

This symphony, an interplay of dimensions, memory, and orchestration, encapsulates the essence of how TensorRT seamlessly converges with C++ for inference. As we explore each note of this symphony, we peel back the layers to reveal the intricate mechanics that underpin the fusion of "TensorRT meets C++."


<br>

#### 1.3. Understanding Memory Operations: Malloc, cudaMalloc, and cudaMemcpy

As we delve into the mechanics of TensorRT and C++ integration, let's illuminate the roles of memory operations—`malloc`, `cudaMalloc`, and `cudaMemcpy`—through clear examples that illustrate their significance in data manipulation.

**1. `malloc`: CPU Memory Allocation**

Our journey begins with `malloc`, a venerable method for memory allocation in C++. This operation reserves memory space on the CPU, where data can be stored and manipulated. But here's the catch—`malloc` operates in the realm of bytes. It expects the size of memory required in bytes. For instance, if we're working with an array of integers, allocating space for 10 integers would involve requesting `10 * sizeof(int)` bytes. This allocated memory is crucial for accommodating data like input and output tensors (`host_input` and `host_output`) within the CPU's memory space.

```cpp
int* host_input = (int*)malloc(10 * sizeof(int));
```
Here, `host_input` becomes an array of 10 integers, ready to hold data in the CPU's memory space.


**2. `cudaMalloc`: Alloacting GPU Memory**

On the GPU's side of the stage, `cudaMalloc` steps in as the protagonist. But there's a twist—`cudaMalloc` needs a pointer on the CPU that stores the address of the allocated memory space on the GPU. This introduces the concept of a pointer to a pointer, often referred to as a double-pointer. Along with the size, you pass the address of the double-pointer where `cudaMalloc` will store the GPU memory's address. This synchronization between the CPU and GPU is pivotal, as it aligns the allocated memory on both sides, preparing for data flow orchestrated by the memory maestro, `cudaMemcpy`.

```cpp
float* device_input;
cudaMalloc((void**)&device_input, 10 * sizeof(int));
```
`device_input` becomes a pointer to GPU memory, reserved for 10 integers.


**3. `cudaMemcpy`: Bridging CPU and GPU**

As the memory symphony crescendos, `cudaMemcpy` takes the conductor's baton. This operation orchestrates the harmonious movement of data between the CPU and GPU. With `cudaMemcpy`, data—like the serialized input image in our journey—travels seamlessly from the CPU's memory (`host_input`) to the GPU's realm (`device_input`). This synchronization ensures that the data exists in both places, primed for the GPU's parallel calculations and the impending inference. Transferring an array of integers from CPU (`host_input`) to GPU (`device_input`):

```cpp
cudaMemcpy(device_input, host_input, 10 * sizeof(int), cudaMemcpyHostToDevice);
```
This synchronizes the data, making `device_input` on the GPU a reflection of `host_input` on the CPU.

With a deepened understanding of these memory operations, we unveil the intricacies of how memory becomes the thread that stitches together the CPU and GPU, forming the cohesive fabric that powers the integration of TensorRT and C++. The mechanics of `malloc`, `cudaMalloc`, and `cudaMemcpy` set the stage for the symphony of operations that unfolds in the TensorRT-powered C++ inference process.

<br>

#### 1.4. Mapping RGB Channels: Aligning Image Data for Inference
In the realm of integrating TensorRT with C++, the subtle yet pivotal process of aligning RGB channels when preparing an image for inference serves as an essential bridge between the image's raw data and the expectations of the TensorRT engine. This alignment transforms the conventional RGB channel order (R1, G1, B1, R2, G2, B2, ...) into a sequential arrangement that TensorRT comprehends (R1, R2, R3, G1, G2, G3, B1, B2, B3), ensuring that the spatial relationships within the image's channels are preserved. The opencv image is stored in memory as `(H,W,C)` while most deep learning based torch models take input image as `(C,H,W)`. Hence, this reordering is needed. This reordering not only ensures the engine's calculations are accurate but also reflects the way neural networks interpret and extract features from images. This seemingly subtle transformation thus becomes a critical step in bridging the gap between image data and the intricacies of deep learning-powered inference.

<div style="width: 80%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_3/ocv_cpp.png" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
 Transforming input image from OpenCV to TensorRT to compensate for (C,H,W) to (H,C,W)
</div>
</div>

<br>

As we wrap up our exploration of these fundamental intuitions, we find ourselves equipped with a solid foundation that bridges the gap between theoretical understanding and practical implementation. From comprehending the journey from a Torch model to C++ inference, to diving into the seamless fusion of TensorRT and C++, understanding memory operations, and aligning image data for optimal inference—we have peeled back the layers that compose the intricate symphony of "TensorRT meets C++."


## 2. Inside the Code 

In this section, we will take a comprehensive journey through the heart of our project, diving deep into the codebase and uncovering its intricacies. Our exploration will be structured into three phases: understanding the project's folder structure, executing the project on your machine, and delving into an in-depth explanation of the crucial code components.
### 2.1 Folder Structure

Here, we unveil the architectural framework that underpins our C++ inference codebase, ensuring a structured and organized approach to integrating TensorRT into the C++ environment. Our code repository encompasses various directories and files, each with a distinct role in facilitating the intricate dance of "TensorRT meets C++."

```plaintext
├── build
├── CMakeLists.txt
├── data
│   ├── hotdog.jpg
│   └── imagenet-classes.txt
├── deploy_tools
│   ├── resnet.engine
│   └── resnet.onnx
├── include
│   ├── inference.hpp
│   ├── postprocessor.hpp
│   └── preprocessor.hpp
├── main.cpp
├── README.md
├── src
│   ├── inference.cpp
│   ├── postprocess.cpp
│   └── preprocessor.cpp
├── tools
│   ├── environment.sh
│   ├── run.sh
│   └── torch_inference.py
└── torch_stuff
    ├── latency.txt
    └── output.txt
```



- **src**: Source code for inference, pre-processing, and post-processing.
- **include**: Header files for communication between project components.
- **deploy_tools**: Serialized TensorRT engine and original ONNX model.
- **data**: Input data like images and class labels.
- **tools**: Utilities for setup, execution, and PyTorch inference.
- **build**: Build artifacts and configuration files.
- **CMakeLists.txt**: Build configuration.
- **main.cpp**: Entry point for the application.
- **README.md**: Comprehensive guide.
- **torch_stuff**: Pytorch Latency and Output 

### 2.2 Project Setup and Execution

To set up and run the project on your machine, follow these steps:

1. Open the `tools/environment.sh` script and adjust the paths for `TensorRT` and `CUDA` libraries as per your system configuration:
    ```bash
    export TensorRT_Lib=/path/to/TensorRT/lib
    export TensorRT_Inc=/path/to/TensorRT/include
    export TensorRT_Bin=/path/to/TensorRT/bin

    export CUDA_Lib=/path/to/CUDA/lib64
    export CUDA_Inc=/path/to/CUDA/include
    export CUDA_Bin=/path/to/CUDA/bin
    export CUDA_HOME=/path/to/CUDA

    export MODE=inference

    export CONDA_ENV=tensorrt
    ```
    Set the `MODE` to `build_engine` for building the TensorRT engine or make it `inference` for running inference on the sample image with the engine.

2. Run the `tools/run.sh` script to execute the PyTorch inference and save its output:

    ```bash
    bash tools/run.sh
    ```

    Upon executing the above steps, you'll observe an informative output similar to the one below, detailing both PyTorch and TensorRT C++ inference results:

    ```plaintext
    ===============================================================
    ||  MODE: inference
    ||  TensorRT: /path/to/TensorRT/lib
    ||  CUDA: /path/to/CUDA
    ===============================================================
    Configuration done!
    =================== STARTING PYTORCH INFERENCE===============================
    class: hotdog, hot dog, red hot, confidence: 60.50566864013672 %, index: 934
    Saved Pytorch output in torch_stuff/output.txt
    Average Latency for 10 iterations: 5.42 ms
    =============================================================================
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /path/to/project/build
    =================== STARTING C++ TensorRT INFERENCE==========================
    class: hotdog, hot dog, red hot, confidence: 59.934%, index: 934
    Mean Absolute Difference in Pytorch and TensorRT C++: 0.0121075
    Average Latency for 10 iterations: 2.19824 ms
    =====================================SUMMARY=================================
    Pytorch Latency: 5.42 ms
    TensorRT in C++ Latency: 2.19824 ms
    Speedup by Quantization: 2.46561x
    Mean Absolute Difference in Pytorch and TensorRT C++: 0.0121075
    =============================================================================
    ```

    With this guide, you can effortlessly set up and run the project on your local machine, leveraging the power of TensorRT in C++ inference and comparing it with PyTorch's results.


### 2.3 Code Deep Dive 
In this section, we'll take a deep dive into the core concepts of running inference with a TensorRT engine in C++. We'll dissect the essential components of our project's codebase and explore how they come together to enable efficient model inference. While we'll primarily focus on the key functions and concepts, we'll also provide an intuitive overview of the supporting functions to ensure a comprehensive understanding of the process.

#### A. Exploring the Inference Class:
The heart of our project lies in the `Inference` class defined in `inference.hpp` and `inference.cpp`, which orchestrates the entire inference process. To understand how the magic happens, we'll focus on the two pivotal functions: `initialize_inference` and `do_inference`. While we'll provide a high-level overview of other functions for context, these two functions encapsulate the most critical aspects of model loading, memory management, and inference execution. Let's break down how these functions work together to achieve accurate and speedy inference results.


**inference.hpp:**

1. **Header Inclusions:** The header includes necessary libraries, such as Nvidia TensorRT headers, OpenCV, CUDA runtime API, and others, for building and running the TensorRT inference engine.

2. **Parameters Struct:** The `Params` struct holds various parameters needed for configuring the inference process, such as file paths, engine attributes, and model parameters.

3. **InferLogger Class:** This class derives from `nvinfer1::ILogger` and is used to handle log messages generated during inference. It's specialized to only print error messages.

4. **Inference Class:** This class encapsulates the entire inference process. It has member functions for building the engine, initializing inference, performing inference, and other helper functions for pre-processing, post-processing, and more.

   - `build()`: Constructs a TensorRT engine by parsing the ONNX model, creating the network, and serializing the engine to a file.
   - `buildFromSerializedEngine()`: Loads a pre-built serialized engine from a file, sets up the TensorRT runtime, and creates an execution context.
   - `initialize_inference()`: Allocates memory for input and output buffers on the GPU, prepares input and output bindings.
   - `do_inference()`: Reads an input image, preprocesses it, populates the input buffer, performs inference, and processes the output to get class predictions.

5. **Helper Functions:** Some inline helper functions are defined for convenience, such as `getElementSize` to determine the size of different data types.

**inference.cpp:**

1. **constructNetwork():** This function is responsible for constructing the TensorRT network by parsing the ONNX model. It configures builder, network, and parser based on user-defined parameters.

2. **build():** This function constructs the TensorRT engine by creating a builder, network, and parser, and then serializing the engine to a file.

3. **buildFromSerializedEngine():** This function loads a pre-built serialized engine from a file, sets up the runtime, and creates an execution context.

4. **read_image():** Reads an input image using OpenCV.

5. **preprocess():** Preprocesses the input image by resizing and normalizing it.

6. **enqueue_input():** Takes the preprocessed image and flattens the RGB channels into the input buffer in a specific order.

7. **initialize_inference():** Allocates GPU memory for input and output buffers, and sets up input and output bindings for the execution context.

8. **do_inference():** Reads an image, preprocesses it, enqueues input, performs inference, calculates latency, and processes the output predictions.

These files encapsulate the core functionality of loading an ONNX model, building a TensorRT engine, performing inference, and processing the results, with necessary pre-processing and post-processing steps. This structure enables you to easily integrate and run the inference process in a C++ environment. Now let us look at the two pivotal functions `initialize_inference()` and `do_inference()` in detail.

**initialize_inference():**

This function is responsible for setting up the necessary memory allocations and configurations required for running inference using the TensorRT engine. Let's break down the code step by step:

1. **Input and Output Buffer Sizes:**
   ```cpp
   int input_idx = mEngine->getBindingIndex("input");
   auto input_dims = mContext->getBindingDimensions(input_idx);
   nvinfer1::DataType input_type = mEngine->getBindingDataType(input_idx);
   size_t input_vol = 1;
   for (int i = 0; i < input_dims.nbDims; i++) {
       input_vol *= input_dims.d[i];
   }
   input_size_in_bytes = input_vol * getElementSize(input_type);
   ```
   This block of code calculates the size of the input buffer based on the input dimensions and data type defined in the TensorRT engine.

2. **Memory Allocation for Input Buffer:**
   ```cpp
   cudaMalloc((void**)&device_input, input_size_in_bytes);
   host_input = (float*)malloc(input_size_in_bytes);
   bindings[input_idx] = device_input;
   ```
   The input buffer is allocated on the GPU using `cudaMalloc`, and corresponding host memory is allocated using `malloc`. The `bindings` array is updated with the device input buffer.

3. **Output Buffer Setup:**
   Similar steps are performed for the output buffer:
   ```cpp
   int output_idx = mEngine->getBindingIndex("output");
   auto output_dims = mContext->getBindingDimensions(output_idx);
   nvinfer1::DataType output_type = mEngine->getBindingDataType(output_idx);
   size_t output_vol = 1;
   for (int i = 0; i < output_dims.nbDims; i++) {
       output_vol *= output_dims.d[i];
   }
   output_size_in_bytes = output_vol * getElementSize(output_type);

   cudaMalloc((void**)&device_output, output_size_in_bytes);
   host_output = (float*)malloc(output_size_in_bytes);
   bindings[output_idx] = device_output;
   ```
   The output buffer size is calculated, memory is allocated on the GPU and host, and the `bindings` array is updated.

**do_inference():**

This function performs the actual inference using the configured TensorRT engine. Let's delve into the detailed explanation:

1. **Read and Preprocess Input Image:**
   ```cpp
   cv::Mat img = read_image(mParams.ioPathsParams.image_path);
   cv::Mat preprocessed_image;
   preprocess(img, preprocessed_image);
   ```
   The input image is read using the `read_image` function, and then preprocessed using the `preprocess` function.

2. **Enqueue Input Data:**
   ```cpp
   enqueue_input(host_input, preprocessed_image);
   cudaMemcpy(device_input, host_input, input_size_in_bytes, cudaMemcpyHostToDevice);
   ```
   The preprocessed image data is enqueued into the input buffer using `enqueue_input`, and then the input data is copied from the host to the GPU using `cudaMemcpy`.

3. **Perform Inference:**
   ```cpp
   bool status_0 = mContext->executeV2(bindings);
   ```
   The inference is executed using the `executeV2` method of the execution context.

4. **Copy Output Data to Host:**
   ```cpp
   cudaMemcpy(host_output, device_output, output_size_in_bytes, cudaMemcpyDeviceToHost);
   ```
   The output data from the GPU is copied back to the host using `cudaMemcpy`.

5. **Calculate Latency:**
   ```cpp
   auto end_time = std::chrono::high_resolution_clock::now();
   std::chrono::duration<float, std::milli> duration = end_time - start_time;
   latency = duration.count();
   ```
   The execution time is calculated to determine the inference latency.

6. **Post-process Output and Classify:**
   ```cpp
   float* class_flattened = static_cast<float*>(host_output);
   std::vector<float> predictions(class_flattened, class_flattened + mParams.modelParams.num_classes);
   mPostprocess.softmax_classify(predictions, verbose);
   ```
   The output data is processed to calculate class predictions using softmax, and the `softmax_classify` function is called from the `mPostprocess` object.


#### B. Exploring the Preprocess and Postprocess Class
In this section, we'll dive into the preprocessor and postprocessor classes. These classes are vital for preparing input data and interpreting model outputs. We'll see how the preprocessor class resizes and normalizes images, while the postprocessor class calculates softmax probabilities and identifies top predicted classes. Understanding these components sheds light on the core data transformations and result analysis in your inference workflow.


**Preprocessor Class:**

The `preprocessor` class handles image preprocessing tasks before feeding them into the neural network for inference. It consists of two main functions: `resize` and `normalization`.

- `resize`:
  This function takes an input image and resizes it to a predefined size using OpenCV's `cv::resize` function. The resized image is stored in the `output_image` parameter. This step ensures that the input image has consistent dimensions that match the requirements of the neural network.

- `normalization`:
  The normalization function standardizes the pixel values of the resized image to be suitable for the neural network. It performs the following steps:
  1. Converts the input image to a floating-point representation and scales pixel values to the range [0, 1].
  2. Subtracts mean values from each channel (RGB) of the image.
  3. Divides the subtracted image by the standard deviation values.
  The normalized image is stored in the `output_image` parameter. These preprocessing steps help ensure that the neural network receives inputs in a consistent format.

**Postprocessor Class:**

Now let's move on to the `postprocessor` class, which handles processing the model outputs after inference.

- `Postprocessor` Constructor:
  The constructor of the `postprocessor` class initializes an instance of the class and accepts a path to a file containing class names. This file is used to map class indices to their human-readable names.

- `softmax_classify`:
  The `softmax_classify` function performs post-processing on the model's output probabilities. It calculates softmax probabilities from the raw output values and prints the top predicted classes along with their confidences. Here's a breakdown of the steps:
  1. The function reads class names from the provided class file and stores them in the `classes` vector.
  2. The softmax probabilities are calculated from the raw output values using the exponential function and normalized.
  3. The function sorts the predicted classes based on their confidences.
  4. If the `verbose` flag is enabled, the top predicted classes with confidences greater than 50% are printed to the console.

Overall, the `postprocessor` class helps interpret the model's output probabilities and provides human-readable class predictions along with confidence values.


#### C. Exploring the Main Code: main.cpp

The `main.cpp` file is the heart of our inference application, where we orchestrate the entire process of building and using the TensorRT engine for inference. Let's break down the key parts of this code to understand how it works.

1. **Command-line Arguments:** The program accepts command-line arguments to determine its behavior. If no arguments are provided or if `--help` is used, a help message is displayed.

2. **Initializing Inference:** The `Params` struct is used to configure various parameters for the inference process. We create an instance of the `Inference` class, passing these parameters.

3. **Command-line Argument Processing:** We iterate through the provided command-line arguments. If `--build_engine` is passed, the TensorRT engine is built using the ONNX model. If `--inference` is passed, the built engine is used for inference.

4. **Performing Inference:** When `--inference` is specified, the following steps occur:
   - We build the TensorRT engine from the serialized engine file.
   - We initialize the inference context and memory buffers.
   - We perform inference using the `do_inference()` method from the `Inference` class.
   - We read the Python-generated output from a file and calculate the mean absolute difference between the Python output and C++ TensorRT output.

5. **Measuring Latency and Speedup:** We measure the average latency of the TensorRT inference over multiple iterations. We also read the Python-generated latency from a file. By comparing the latencies, we calculate the speedup achieved by the TensorRT inference.

6. **Displaying Summary:** Finally, we display a summary of the results, including the PyTorch latency, TensorRT latency, speedup, and mean absolute difference in predictions.

This `main.cpp` file demonstrates how we can effectively build, deploy, and analyze the performance and accuracy of our TensorRT-based inference system.


#### D. Building the Project with CMake: CMakeLists.txt

The `CMakeLists.txt` file is crucial for configuring the build process of your project. It defines how to compile the code and link various libraries. Let's dive into the key components of this CMake configuration.

1. **Minimum Required Version:** Specify the minimum required CMake version for your project.

2. **Project Configuration:** Set the project name and C++ standard version.

3. **Setting RPATH:** Configure the runtime path for your executable using `$ORIGIN`.

4. **Include Directories:** Define the include directories for your project, including OpenCV and CUDA.

5. **Find OpenCV:** Use the `find_package` command to locate and configure OpenCV.

6. **Find CUDA:** Use the `find_package` command to locate and configure CUDA.

7. **Enable CUDA Support:** Set the CUDA compiler and flags to enable CUDA support.

8. **Linking Preprocessor and Postprocessor Libraries:** Build and link the `pre_processor` and `post_processor` libraries. These libraries contain the preprocessing and postprocessing logic.

9. **Building the Inference Library:** Create the `Inference` shared library by compiling and linking the `inference.cpp` file. This library is used to perform inference using TensorRT.

10. **Linking Dependencies:** Link the `Inference` library with the `pre_processor`, `post_processor`, and necessary libraries such as `nvinfer`, `nvonnxparser`, `stdc++fs`, CUDA libraries, and OpenCV.

11. **Creating the Executable:** Build the `main` executable, which serves as the entry point for your application. Link it with the `Inference` library.

This `CMakeLists.txt` file defines how your project will be built, linking various libraries and ensuring proper configuration for successful compilation and execution.


## 3. Latency and Consistency

In this section, we will delve into the results obtained from our TensorRT-powered inference and compare them with PyTorch. The performance improvements achieved through TensorRT are significant, showcasing both speedup and consistency in inference latency.

### 3.1 Speedup by Quantization

Quantization is a technique that reduces the memory and computational requirements of neural network models. When comparing PyTorch's latency to that of our C++ TensorRT-powered model, we observe a remarkable speedup. PyTorch exhibits a latency of 5.42 ms, while the C++ TensorRT combination achieves an impressive latency of 2.19824 ms. This translates to a speedup of approximately `2.46561 times.`

### 3.2 Minimal Output Difference

Although the inference latencies have changed, it's crucial to ensure that the actual outputs remain consistent between the original PyTorch model and the C++ TensorRT implementation. Fortunately, our analysis reveals that the Mean Absolute Difference between the two is merely `0.0121075`. This minor variation in output demonstrates the reliability and accuracy of the TensorRT-powered inference.

The combined benefits of reduced latency and minimal output differences make the integration of TensorRT into the project a powerful optimization, ensuring efficient and consistent real-time inferencing for various applications.

## 4. Conclusion

In this project, we embarked on a journey to optimize neural network inference using TensorRT, a high-performance deep learning inference optimizer and runtime library. By integrating TensorRT into our C++ application, we achieved remarkable improvements in inference speed and consistency, enhancing the overall efficiency of our model.

Throughout the exploration, we dissected the intricacies of the TensorRT engine, delving into core concepts such as building, initializing, and executing the engine for both PyTorch and C++ implementations. We gained insights into preprocessing and postprocessing techniques to ensure accurate input and output handling. Our journey was enriched by understanding the integration of CUDA and OpenCV libraries, which are essential for seamless GPU acceleration and image processing.

By combining the power of TensorRT, CUDA, and C++, we witnessed a substantial reduction in inference latency. The speedup achieved—showcased through a quantified speedup factor—highlighted the importance of optimizing deep learning models for real-time applications. Additionally, our evaluation revealed a minimal Mean Absolute Difference between the outputs of PyTorch and the C++ TensorRT implementation, affirming the reliability of our optimization strategy.

In conclusion, our project underscores the significance of leveraging TensorRT in tandem with C++ for neural network inference. This integration paves the way for enhanced performance, making it a pivotal solution for real-time applications across various domains. Through this exploration, we've gained valuable insights into the world of deep learning optimization, setting the stage for further innovations in the field.

