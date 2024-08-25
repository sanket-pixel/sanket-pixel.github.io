---
layout: post
title: Down the CudaMemory lane
date: 2024-07-14 15:53:00-0400
description: Data Transfers Between CPU and GPU 
thumbnail : /assets/img/blog/blog_5/memory.png
categories: quantization
tag : [cuda, nvidia]
giscus_comments: false
related_posts: true
---

#### In the fast-paced landscape of deep learning deployment, where performance optimization is critical, understanding the foundational principles behind memory management and data transfer between CPUs and GPUs is essential. This blog aims to demystify these concepts by starting from first principles. We'll explore how memory allocation operates on CPUs using `malloc` and on GPUs using `cudaMalloc`, shedding light on their distinct functionalities. Additionally, we'll unravel the complexities of `cudaMemcpy`, a crucial function for seamless data exchange between these processing units. By clarifying these fundamental concepts, we empower developers to strategically optimize their applications for maximum efficiency and speed in GPU-accelerated environments.

<br>
<div style="width: 95%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_5/memory.png" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
 Data Transfers Between CPU and GPU 
</div>
</div>

We will understand the fundamentals of CPU and GPU memory manipulations and writing and reading data from the same in the following sections. We first look at how to allocate memory on CPU and GPU, following which we look at how to copy data from the CPU to the GPU and vice versa.


### 1. Understanding Memory Allocation
Efficient memory allocation is fundamental to optimizing computational tasks, particularly when working with CPU and GPU resources. This section delves into the specifics of `malloc` and `cudaMalloc`, the primary functions used for memory allocation in C++ and CUDA respectively, and highlights the key differences between CPU and GPU memory allocation.

#### 1.1 malloc
`malloc`, short for "memory allocation," is a standard library function in C and C++ used to allocate a block of memory on the heap. The memory allocated by malloc is uninitialized and is typically used for dynamic data structures such as arrays and linked lists. 

<div style="width: 80%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_5/malloc.png" title="malloc" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Malloc
</div>
</div>


**Example Usage**

In this example, we allocate memory for an array of integers of size 10. 
```cpp
size_t size = 10 * sizeof(int); // Allocate memory for 10 integers
```
We first compute the size of the memory that needs to be allocated on the CPU in bytes, since the malloc function accepts the size parameter in bytes. To that end, we multiply the desired array size ( 10 in this case ) with the bytesize of the desired data type ( integer in this case ). The `sizeof` function returns the size of the datatype in bytes. Just for information, this results in 40 ( 10 * 4) since each integer has the size of 4 bytes.

Next, we use the all important `malloc()` function to allocate the memory of the size computed before.

```cpp
int *array = (int*) malloc(size);
```
The malloc function returns a pointer pointing to the memory address in the CPU where the allocation has been done. In particular, it returns a `void*` type pointer. Since we are trying to create an array of integers, we type case this `void*` to `int*`. 

A good question that should arise at this point is, what difference does this typecasting have on the actual CPU memory. The memory block allocated by `malloc` is simply a contiguous block of raw bytes. The casting does not alter this raw memory. The casting simply lets the compiler know that this memory address points to an array of integers. The advantages are as follows :

**a. Pointer Arthematic**
 
 With an `int*` pointer, pointer arithmetic operates in units of `int`. For example, if array is an `int*`, then array + 1 points to the next int in the array (4 bytes away if sizeof(int) is 4).

**b. Assigning Elements**

After type typecasting to `int*` integer values can be assigned by simply passing indexm like a regular integer array.
```cpp
for (int i = 0; i < 10; ++i) {
    array[i] = i * i; // Store the square of i in the allocated array
}
```

**c. Accessing Elements**

`array[i]` accesses the i-th int in the allocated memory block.

```cpp
for (int i = 0; i < 10; ++i) {
    std::cout << array[i] << " "; // Print the values stored in the array
}
```
To summarize, here is the code for allocating memory on the CPU for an array of 10 integers.

```cpp
size_t size = 10 * sizeof(int); // Allocate memory for 10 integers
int *array = (int*) malloc(size);
if (array == nullptr) {
    std::cerr << "Failed to allocate memory" << std::endl;
    return EXIT_FAILURE;
}
// Use the allocated memory
free(array); // Free the allocated memory when done

```
#### 1.2 cudaMalloc
`cudaMalloc` is a function provided by CUDA for allocating memory on the GPU. This function is analogous to malloc but is specifically designed for GPU memory, allowing developers for accessing and manipulating GPU memory.

<div style="width: 80%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_5/cudamalloc.png" title="malloc" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
CudaMalloc
</div>
</div>

**Example Usage**

Let us stick with the same example as we used for malloc(), but this time we allocate the array of integers of size 10 on the GPU.
For starters, just like before, lets compute the number of bytes needed for an integer array of size 10.

```cpp
size_t size = 10 * sizeof(int); // Allocate memory for 10 integers on GPU
```
Now, we declare a pointer to the start of this integer array, just like we would for the CPU. 

```cpp
int *d_array;
```
But there is a small catch here. 

***Alhthough we declare the `int *d_array` on the CPU, this pointer is intended to point to a memory location on the GPU.***

We will understand this in further detail, once we complete the memory allocation on the GPU.
To that end, we now add the final missing piece which is calling `cudaMalloc()`

```cpp
cudaError_t err = cudaMalloc((void**)&d_array, size);
```

As shown above, the `cudaMalloc()` function, expects **a pointer to a pointer** as input. In other words, we first declare a device pointer `int *d_array`, and then pass the address of this pointer on the CPU `(&d_array)` to the cudaMalloc function.
We then cast this pointer to a pointer to type `(void**)` and pass that as input along with the size of intended memory allocation in bytes.

This function then allocates the expected size of memory onto the GPU starting from memory location `d_array`. Let us understand this visually.

<div style="width: 80%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_5/deviceptr.png" title="malloc" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Intuition of a Device Pointer. The CPU location 4 stores the allocated GPU memory location C.
</div>
</div>

In this example, for the sake of simplicity, lets assume that the CPU memory address is indexed by numbers 1,2,3... and so on. 
Furthermore, the GPU is indexed by A,B,C.. and so on.  In the first step, when `int *d_array` is declared, the memory location 4 on the CPU
is allocated, to store the address of the GPU. 

Then after `cudaMalloc`, this address stores the value C, which is the memory location allocated on the GPU.
In this context, `*d_array` is **C** (on the GPU) and `&(*d_array)` is **4** ( on the CPU).
To summarize, here is the code for allocating memory for 10 integers on the GPU.

```cpp
size_t size = 10 * sizeof(int); // Allocate memory for 10 integers on GPU
int *d_array;
cudaError_t err = cudaMalloc((void**)&d_array, size);
if (err != cudaSuccess) {
    std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
    return EXIT_FAILURE;
}
```

Now that we have understood memory allocation, lets look at how to copy data within the CPU, GPU and between CPU and GPU.

### 2. Copying explained from first principals

Now that we have understood memory allocation on both the CPU and GPU, let us understand how to transfer data between them.
We will use the same example of an integer array of size 10, to keep life simple, and focus on understanding the underlying concepts from first principals.

In this example, we will perform the following operations :

1. Allocate memory on the CPU for integer array of size 10.
2. Allocate memory on the GPU for integer array of size 10.
3. Initialize the CPU array with squares of integers from 1 to 10.
4. Copy data from the CPU to the GPU.
5. Copy the data back from the GPU to the CPU.
6. Print values to verify it all works.


Lets look at each step, one at a time.

#### STEP 1 : Allocate memory on the CPU

As already explained, we first compute the size in bytes and allocate appropriate memory using `malloc()`.

```cpp
size_t size = 10 * sizeof(int);
// Allocate memory on the CPU (host)
int *h_array = (int*) malloc(size);
if (h_array == nullptr) {
    std::cerr << "Failed to allocate CPU memory" << std::endl;
    return EXIT_FAILURE;
}
```

#### STEP 2 : : Allocate Memory on the GPU
We first compute the size of allocation in bytes and use `cudaMalloc` as explained above.
```cpp
// Allocate memory on the GPU (device)
int *d_array;
cudaError_t err = cudaMalloc((void**)&d_array, size);
if (err != cudaSuccess) {
    std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
    free(h_array); // Free CPU memory if GPU allocation fails
    return EXIT_FAILURE;
}
```

#### Step 3: Initialize the CPU Array

We initialize the CPU array with the squares of integers from 1 to 10.

```cpp
// Initialize CPU memory with data
for (int i = 0; i < 10; ++i) {
    h_array[i] = (i + 1) * (i + 1); // Squares of integers from 1 to 10
}
```

#### Step 4: Copy Data from the CPU to the GPU
We now copy this data from the CPU to GPU in the allocated memory.
Here we take a small stop, and first understand `cudaMemcpy` in detail.

##### 4.1 cudaMemcpy Explained
The `cudaMemcpy` function is essential for transferring data between the `host` (CPU) and the `device` (GPU) in CUDA programming. Let's break down this function and understand it intuitively from first principles. First, the syntax is as shown in the figure below.

<div style="width: 80%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_5/cudamemcpy.png" title="malloc" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Syntax of cudaMemcpy.
</div>
</div>

The function takes in the destination pointer ( the memory location where the data needs to be copied to), source pointer (the memory location from where the data needs to be copied from ). Furhtermore, it also takes in the number of bytes that need to copied from. It is important to note, that these destination and source pointers may be on the CPU and GPU, depending on the direction of copying provided in the final parameter `cudaMemcpyKind`.
It is pretty obvious that there can only be 4 possible direction of copying data in this context.

1. `cudaMemcpyHostToHost`: Copy data from host to host.
2. `cudaMemcpyHostToDevice`: Copy data from host to device.
3. `cudaMemcpyDeviceToHost`: Copy data from device to host.
4. `cudaMemcpyDeviceToDevice`: Copy data from device to device.

Now that we have the syntax and intuition of `cudaMemcpy` covered, let us continue with our example and use this function.

```cpp
// Copy data from CPU to GPU
err = cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from CPU to GPU: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_array); // Free GPU memory if copy fails
    free(h_array);     // Free CPU memory
    return EXIT_FAILURE;
}
```
Here, we use `cudaMemcpy` to copy the data from the CPU (`h_array`) to the GPU (`d_array`). We specify `cudaMemcpyHostToDevice` to indicate the direction of the copy.

#### Step 5:  Copy Data from the GPU to the CPU
We now copy the data back to the CPU from the GPU. As expected the source pointer now becomes the GPU pointer and destination becomes the CPU pointer.
Furthermore the direction of copying is from device to host so the `cudaMemcpyKind` will be `cudaMemcpyDeviceToHost`.

```cpp
err = cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from GPU to CPU: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_array); // Free GPU memory if copy fails
    free(h_array);     // Free CPU memory
    return EXIT_FAILURE;
}

```

#### Step 6 : Verify consistency after copying

Just for completeness we now verify if the data copied back from the GPU to CPU is as expected by simply printing the host array.

```cpp
// Print the data copied back from GPU to CPU
for (int i = 0; i < 10; ++i) {
    std::cout << h_array[i] << " ";
}
std::cout << std::endl;

// Free the allocated memory
cudaFree(d_array);
free(h_array);

```

In the end, we free the GPU and CPU memory using `cudaFree` and `free` respectively,

Let us look at the entire code to recap. 

```cpp
#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Step 1: Allocate Memory on the CPU for an Integer Array of Size 10
    size_t size = 10 * sizeof(int);

    // Allocate memory on the CPU (host)
    int *h_array = (int*) malloc(size);
    if (h_array == nullptr) {
        std::cerr << "Failed to allocate CPU memory" << std::endl;
        return EXIT_FAILURE;
    }

    // Step 2: Allocate Memory on the GPU for an Integer Array of Size 10
    int *d_array;
    cudaError_t err = cudaMalloc((void**)&d_array, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
        free(h_array); // Free CPU memory if GPU allocation fails
        return EXIT_FAILURE;
    }

    // Step 3: Initialize the CPU Array with Squares of Integers from 1 to 10
    for (int i = 0; i < 10; ++i) {
        h_array[i] = (i + 1) * (i + 1); // Squares of integers from 1 to 10
    }

    // Step 4: Copy Data from the CPU to the GPU
    err = cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy data from CPU to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_array); // Free GPU memory if copy fails
        free(h_array);     // Free CPU memory
        return EXIT_FAILURE;
    }

    // Step 5: Copy the Data Back from the GPU to the CPU
    err = cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy data from GPU to CPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_array); // Free GPU memory if copy fails
        free(h_array);     // Free CPU memory
        return EXIT_FAILURE;
    }

    // Step 6: Print Values to Verify It All Works
    for (int i = 0; i < 10; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // Free the allocated memory
    cudaFree(d_array);
    free(h_array);

    return EXIT_SUCCESS;
}
```

### 3. Conclusion

In this blog, we explored the essential concepts of memory allocation and data transfer between the CPU and GPU, which are key to optimizing performance in GPU-accelerated applications. We began by examining how memory is allocated on the CPU using malloc, which reserves a block of memory for use in programs. We then discussed GPU memory allocation with cudaMalloc, which functions similarly but is used for allocating memory on the GPU.

Following memory allocation, we demonstrated how to transfer data between the CPU and GPU using cudaMemcpy. We highlighted the process of copying data from the CPU to the GPU, performing computations, and then copying the results back to the CPU. This seamless data transfer enables efficient use of the GPU's processing power, allowing for faster and more powerful computations. Understanding these fundamental operations is crucial for anyone looking to leverage GPU acceleration in their applications.









