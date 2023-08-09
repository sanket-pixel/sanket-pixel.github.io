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


Welcome to our blog series where the worlds of TensorRT and C++ converge to revolutionize the AI landscape. In our previous installment, "Have you met TensorRT?," we embarked on an exciting exploration of the fundamental concepts, laying the groundwork for the cutting-edge journey we're about to embark upon. Now, with "TensorRT meets C++," we usher you into a realm of possibilities where the seamless integration of these technologies has profound implications for AI, particularly in the context of robotics and edge computing.

#### Unveiling the Power of Integration

The significance of this integration cannot be understated. While our prior post introduced you to TensorRT's prowess in Python, this blog takes you a step further. By blending TensorRT with the power of C++, we equip you with the skills to transition your PyTorch models seamlessly into a C++ environment. This transition isn't just about speed—although the enhanced inference speed is undoubtedly thrilling—it's about more. It's about delving into the heart of memory management, understanding the intricacies of processor operations, and acquiring a deeper comprehension of how your models interact with the hardware.

#### The Challenge of Copying: Navigating the GPU Highway

With great power comes great responsibility. The marriage of TensorRT and C++ introduces a pivotal challenge: the accurate transfer of data between the CPU and GPU. As we embark on this journey, we delve into the problem of copying data correctly onto the GPU and navigating the intricacies of memory transfers. We dissect this challenge, peeling back the layers to understand how to harmonize these distinct processing units and deliver seamless data flow.

#### Unveiling a New Frontier: From Vision to Robotics

Our blog isn't just about one aspect of AI—it's about a journey that spans diverse domains. This integration is the key that unlocks possibilities beyond image classification. From lidar-based perception to planning-based decision-making, from text-based sentiment analysis to complex deep reinforcement learning, the door to countless applications opens. All these, in the realm of robotics and edge AI where C++ reigns supreme. As you delve into this blog, you're not merely mastering a technology; you're bridging the chasm between your Jupyter notebooks and the robotics that can wield your models.

#### The Road Ahead: Sections to Explore

As we journey through "TensorRT meets C++," we'll traverse several vital sections:

1. **Intuition**: Lay the groundwork by understanding the synergy between TensorRT and C++.

2. **How to Copy Stuff Properly?**: Tackle the challenge of efficient data copying, mastering the art of smooth GPU transitions.

3. **Inside Deep Dive**: Embark on an exploration of the inner workings, understanding the harmony of memory and processing.

4. **Latency Measurement**: Quantify the gains—measure the reduced latency that TensorRT offers.

5. **Consistency with PyTorch Output**: Bridge the gap by ensuring consistent output between your PyTorch and C++ TensorRT models.

6. **Conclusion**: Weave together the threads of knowledge, leaving you ready to wield the integration with confidence.

Prepare to witness the fusion of AI and robotics—a fusion that empowers you to take your models beyond the notebook and into the real world, where C++ is the language of choice. Let's embark on this transformative journey together.


## 1. Intuition
In the journey of merging the powerful capabilities of TensorRT with the versatile landscape of C++, we embark on a comprehensive exploration of intuitive concepts that form the bedrock of seamless model inference. This section of the blog delves into four key intuitions, each unraveling a layer of understanding that enriches our grasp of the integration process. From traversing the path from a Torch model to C++ inference to uncovering the mechanics where TensorRT and C++ converge, and from unraveling the intricacies of memory operations to mapping RGB channels for perfect alignment—the intuitions explored herein shed light on the intricate dance between theory and application. These insights, rooted in both practical implementation and theoretical foundations, propel us toward mastering the harmonious symphony that is "TensorRT meets C++."

####  A. From Torch Model to C++ Inference

In the realm of AI and machine learning, the journey from a PyTorch model within the cozy confines of a Jupyter notebook to real-world applications demands a bridge—a bridge that harmonizes the worlds of deep learning and robust, high-performance C++ execution. Enter TensorRT, the catalyst that transforms this transition from an ambitious leap to a smooth stride. Let us look at an overview of how we make this transition.
 
**Step 1: The Torch-to-ONNX Conversion**

At the heart of this transformation lies the conversion of the PyTorch model into the Open Neural Network Exchange (`ONNX`) format. This universal format serves as the lingua franca between frameworks, unlocking the potential to bridge the gap between PyTorch and TensorRT. The Torch-to-ONNX conversion encapsulates the model's architecture and parameters, setting the stage for seamless integration into the C++ landscape.

**Step 2: The ONNX-to-TensorRT Metamorphosis**

With the ONNX representation in hand, in form of an `.onnx` file, we traverse the second leg of our journey—the ONNX-to-TensorRT transformation. Here, the ONNX model metamorphoses into a high-performance TensorRT engine, optimized to harness the prowess of modern GPUs. TensorRT's meticulous optimization techniques prune the neural network, leveraging the inherent parallelism of GPUs for expedited inference without compromising accuracy. This can happen via the `trtexec` tool provided by TensorRT, via the Python API or through the C++ API. Having already covered the Python API implementation in the previous blog, we shall see the C++ API execution later in this blog.

**Step 3: Unveiling C++ Inference**

And now, with the TensorRT engine prepared, in form of a `.engine` file, we navigate the final stretch of our voyage—the integration of the engine into C++ for inference. Armed with the TensorRT-powered engine, C++ becomes the stage where our models perform with astounding efficiency. Leveraging C++'s capabilities, we create a pipeline to read input data, offload it onto the GPU, execute inference, and retrieve the output—effortlessly spanning the distance between deep learning models and real-world applications.

The diagram below captures the essence of this journey, illustrating how the torch model transforms into an optimized TensorRT engine and finds its home in the world of C++.

<!-- Insert your diagram here -->

<br>


#### B. The Fusion: Where TensorRT Meets C++

In the intricate choreography between TensorRT and C++, a harmonious integration unfolds, paving the way for seamless model inference. As we transition from a TensorRT engine to the dynamic landscape of C++, a meticulous orchestration of steps ensures a blend of precision, speed, and efficiency that bridges the gap between these two powerful realms.

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

#### C. Understanding Memory Operations: Malloc, cudaMalloc, and cudaMemcpy

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

#### D. Mapping RGB Channels: Aligning Image Data for Inference
In the realm of integrating TensorRT with C++, the subtle yet pivotal process of aligning RGB channels when preparing an image for inference serves as an essential bridge between the image's raw data and the expectations of the TensorRT engine. This alignment transforms the conventional RGB channel order (R1, G1, B1, R2, G2, B2, ...) into a sequential arrangement that TensorRT comprehends (R1, R2, R3, G1, G2, G3, B1, B2, B3), ensuring that the spatial relationships within the image's channels are preserved. This reordering not only ensures the engine's calculations are accurate but also reflects the way neural networks interpret and extract features from images. This seemingly subtle transformation thus becomes a critical step in bridging the gap between image data and the intricacies of deep learning-powered inference.



As we wrap up our exploration of these fundamental intuitions, we find ourselves equipped with a solid foundation that bridges the gap between theoretical understanding and practical implementation. From comprehending the journey from a Torch model to C++ inference, to diving into the seamless fusion of TensorRT and C++, understanding memory operations, and aligning image data for optimal inference—we have peeled back the layers that compose the intricate symphony of "TensorRT meets C++."