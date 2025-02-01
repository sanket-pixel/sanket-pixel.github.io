---
layout: post
title: That First CUDA Blog I Needed
date: 2024-10-20 10:53:00-0400
description: The ideal first blog to start learning CUDA.
thumbnail : /assets/img/blog/blog_8/cuda.png
categories: cuda
tag : [nvidia, cuda]
giscus_comments: false
related_posts: true
---

#### In this blog, we’re going to dive into one of the most critical concepts in CUDA programming: shared memory. Shared memory is like the secret ingredient that can supercharge your GPU code. While CUDA’s global memory serves as the main storage, it’s often slow to access repeatedly. That’s where shared memory comes in. It acts as a customizable, fast-access scratchpad where you can store data that is frequently reused by threads within the same block, helping you avoid costly memory transfers. We’ll explore how this works, why it matters, and how you can use it to make your CUDA programs much faster. 
<br>

<div style="width: 80%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/cuda.png" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   Let us begin the CUDA journey, the right way.
</div>
</div>


<!-- ``` ### 0. Prerequisites to learn CUDA``` -->

### 1. Hello World
A good way to learn something new, is to begin from something you already know and then connect the dots. Let us first look at a simple `Hello World` program in C++. 

```cpp
#include <iostream>
int main(){
    std::cout << "Hello World!" << std::endl;
}
```

To execute this program on your machine, follow the following steps :

1. Open a terminal and navigate to the directory containing this `.cpp` file. 
    ```
    cd that_first_cuda_blog/1_hello_world
    ```

2. Compile the program using the following command 
    ```bash
    g++ hello_world.cpp -o hello_world
    ```
3. This will create an executable `hello_world` in this directory. Execute it using 
    ```
    ./hello_world
    ```

This should print `Hello World!` in the terminal, as expected. Here, the **g++ compiler** compiles the source code `hello_world.cpp` and translates it to machine code, in form of an executable file. 
The CPU then executes this machine code, to print `Hello World!` onto the temrinal. 

If one intends to execute the same on an NVIDIA GPU, **CUDA** can be used.<br>
CUDA is a programming framework, that allows programmers to talk to NVIDIA GPUs via the CPU.

 **TODO : TALK HERE ABOUT HOW CUDA HAS PARALLILAZATION USING THREADS**

Let us look at simple Hello World example in CUDA. 

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void gpu_hello_world(){
  printf("Hello World from GPU! \n");
}

int main(){
  std::cout << "Hello World from CPU!" << std::endl;
  gpu_hello_world<<<1,1>>>();
  cudaDeviceSynchronize();
}
```

To execute this program on your machine, follow the following steps :

1. Open a terminal and navigate to the `1_hello_world` directory 
    ```
    cd that_first_cuda_blog/1_hello_world
    ```

2. Compile the program using the following command 
    ```bash
    nvcc hello_world_gpu.cu -o hello_world_gpu
    ```
3. This will create an executable `hello_world_gpu` in this directory. Execute it using 
    ```
    ./hello_world_gpu
    ```
4. The output should be the following 
    ```
    Hello World from CPU!
    Hello World from GPU! 
    ```

Now that we have our first CUDA program running, let us dissect this CUDA program, and understand how it works from first principles. <br>

```cpp
__global__ void gpu_hello_world(){
  printf("Hello World from GPU! \n");
}
```

The code snippet above is a function that is intended to run on the GPU. <br> 
In the GPU jargon, such a function is called **kernel**. <br>

Kernel, specifically, is a special function, that can be invoked from the CPU, but runs only on the GPU.
CPU, is generally referred to as `host` and GPU is referred to as `device`, since the CPU hosts the GPU in some sense.
The `__global__` keyword is used to specify that this function is a **kernel**, in that, it can be called from the host but executed on the device.


`gpu_hello_world<<<1,1>>>();` is a CUDA-specific syntax. We will discuss what `<<<1,1>>>` means later in this blog. 
For now it is sufficient to understand that `<<<1,1>>>`, allocates 1 thread for executing this kernel. 

Let us first understand how this kernel launch works from first principles.

1. The `host` (CPU) executes instructions ( compiled lines of code ), one at a time, sequentially. 
2. When it reaches the kernel launch instruction (`gpu_hello_world<<<1,1>>>();`), the host launches the kernel.
Under the hood, the **CUDA Runtime Library** on the host, places the launch command onto a **CUDA Stream** which a queue mantained 
on the host.  This queue is designed to hold kernel launches, memory transfer requests and other CUDA tasks, to ensure they execute sequentially for the same **CUDA Stream**.
We will dissect  **CUDA Streams** later in this blog. 
3. The CUDA Runtime, now hands over the launch commands to the **NVIDIA Driver** on the `host`, which is responsible for talking to the `device` (GPU).
4. The **NVIDIA Driver** pushes this launch command to the **command buffer** which is managed by the GPU hardware. This buffer resides on the `device` and 
holds the commands to be executed, once sufficient GPU resources are available.
5. The GPU, once resources are available, pulls commands from the **command buffer** and starts executing them.
6. The **host does not wait for the kernel execution to finish**, and moves on with the next instruction. This execution approach is known as **asynchronous**. In particular, the `host` and `device` executes independently and simulatenously. When a command is like kernel launch is issued by the host, it does not wait for the command to complete on the `device`, but simply moves on to the next instruction, while the `device` handles the requested operation in parallel.

To unerstand this better, we can change the earlier `hello_world.cu` source code, by commenting out `cudaDeviceSynchronize();`.

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void gpu_hello_world(){
  printf("Hello World from GPU! \n");
}

int main(){
  std::cout << "Hello World from CPU!" << std::endl;
  gpu_hello_world<<<1,1>>>();
  // comment out this line. 
  // Now the host does not wait for the device and moves on.
  // cudaDeviceSynchronize();
}
```

The output of this program will be just as follows :

```
Hello World from CPU!
```

Note, that since we removed `cudaDeviceSynchronize();`, the host launches the `gpu_hello_world` kernel and moves on to the next instruction. The exection of the host code finishes, even before the `device` completes, hence it does not print `Hello World from GPU!` onto the output buffer.

Let us now extend our single thread CUDA Hello World, to run it with 8 threads. We would like the GPU to repeat this same "Hello World from GPU" operation 8 times. Just one small change in our original code will make this happen.


```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void gpu_hello_world(){
  printf("Hello World from GPU! \n");
}

int main(){
  std::cout << "Hello World from CPU!" << std::endl;
  // HERE , we replace <<<1,1>>> with <<<1,8>>>.
  gpu_hello_world<<<1,8>>>();
  cudaDeviceSynchronize();
}
```

The output of this code, will be one `Hello World from CPU!` and 8 `Hello World from GPU!`s. 
The main change as explained in the comment above the kernel code is replace `<<<1,1>>>` with `<<<1,8>>>`, which essentially
means launching the same kernel with 8 threads. The GPU runs 8 "print Hello World" operations in parallel. 

We will understand what `<<<1,8>>>` exactly means in absolute detail, but at this point, it is sufficient to understand that `<<<1,1>>>`
launches one thread and `<<<1,8>>>` launches 8 threads in parallel. 

In summary, in this **Hello World** section, we first looked at how to print Hello World using the CPU, followed by the same using the GPU.
The major takeaway from this section is to understand what are kernels in general, and how *exactly* is a kernel launched from the `host`, to run the same operations in parallel on the `device`.

<!-- TODO : ADD OS Concepts here -->

### 2. Print Square of Numbers
The basic foundation is now laid and we will now lay some more foundation on top.
Let us print square of list of integers. 

```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 5
__global__ void print_square(){
  unsigned id = threadIdx.x;
  printf("%d\n", id * id);
}

int main(){
  for(int i = 0; i<N;i++){
    std::cout << i * i << std::endl;
  }
  print_square<<<1,N>>>();
  cudaDeviceSynchronize();
}
```










