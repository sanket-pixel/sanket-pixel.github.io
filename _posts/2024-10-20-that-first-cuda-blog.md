---
layout: post
title: That First CUDA Blog I Needed
date: 2024-10-20 10:53:00-0400
description: The ideal first blog to start learning CUDA.
thumbnail : /assets/img/blog/blog_8/kindergarten.jpeg
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
        {% include figure.html path="/assets/img/blog/blog_8/kindergarten.jpeg" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   Let us begin the CUDA journey, the right way.
</div>
</div>


#### **1. The paradigm shift from CPU to GPU World**

A GPU is not just a faster CPU. It is rather, a hardware designed specifically for doing ***similar things, a lot of times, all at once***. 
When one goes from programming on the CPU to the GPU, the mindset shifts from `"How do I do this?"` to `"How do I do this with a 1000 workers?"`.
The most challenging part in the beginnning is not the CUDA syntax or the intricacies of the GPU memory. But it is the shift from thinking serially or sequentially 
to thinking in parallel.

This shift requires you to stop thinking like a single worker doing one task after another, and start thinking like a supervisor assigning the same task to a massive team—each working on a different part of the problem at the same time. You no longer write instructions for the whole job; instead, you write instructions for one worker and trust the system to repeat it across thousands. The challenge is learning to break down problems into small, identical tasks that can run side by side—without depending on each other or getting in each other’s way.


<br>
<div style="width: 80%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/cpu_vs_gpu.jpeg" title="cpugpu" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   CPU: The lone chef. GPU: The parallel kitchen team.
</div>
</div>

Let’s say we want to multiply each number in an array by 2.  

On the CPU, you would think sequentially as shown below :

``` python
for i from 0 to N-1:
    output[i] = input[i] * 2
```

Here, you think like a `single worker` walking through the entire list, `one element at a time`.  
While on the GPU you must think of processing in parallel as shown below :
```python 
function worker(i):
    output[i] = input[i] * 2

launch N workers:
    each runs worker(i) with its own i
```
Instead of one worker doing the whole loop, you write code for just `one worker`, and let `thousands of them` each handle their own `i` independently.
You’re no longer in control of the whole process—you’re only describing what one tiny part of the system should do. This mental shift—from controlling a loop to writing instructions for an army of workers—is what makes parallel thinking hard at first.


#### **2. Groundwork: What CUDA Assumes You Know**
If you’ve spent most of your time in languages like Python, JavaScript, or even high-level C++ without touching low-level memory concepts, CUDA will feel different. That's because CUDA code is almost always written in C or C++, and runs in an environment where you’re much closer to the hardware. Let us understand some core low level programming concepts you should know, before finally writing our first CUDA kernel in the next section.

##### **2.1 Pointers : Variables that point to other Variable**
A pointer is a variable that stores the memory address of another variable. In Python, you deal with lists and objects without thinking about where they live in memory. But in CUDA (and C/C++), you often work with memory addresses directly. In this fighure below, `x` is an integer variable that holds the value `10`, and it lives at memory address `0x1234`.
<br>
<div style="width: 50%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/pointer.png" title="pointer" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   Illustration of a pointer in C++ — ptr stores the memory address of variable x, allowing indirect access to its value.
</div>
</div>

The pointer `ptr` is declared to hold the address of an integer. When we assign it `&x`, we're storing the address of `x` in `ptr`. So now `ptr` contains `0x1234` — it points to `x`.
```c
int x = 10;
int* ptr = &x;  // ptr now holds the memory address of x (e.g., 0x1234)
```
The pointer itself lives at a different memory location, say `0x1550`. If we access `*ptr`, we get the value stored at the address it points to — in this case, `10`. And if we use `&ptr`, we get the address where the pointer itself is stored — `0x1550`.

##### **2.2 Functions : Parameter Passing by Value vs. Reference**
A function is a reusable block of code that performs a specific task and can take inputs (parameters) and return outputs — just like how functions work in Python.
In C, when you `pass by value`, the function gets a `copy of the data`. When you `pass by reference` (using a pointer), the function gets access to the `original`, so it can modify it.
```c
void modify(int x);     // gets a copy of x
void modify(int* x);    // gets the original x via address
```

Passing by value is like giving someone a photocopy of a document, while passing by reference is like giving them the original paper to make changes on.

##### **2.3 Arrays and Memory Layout**
In high-level languages, arrays feel like magical lists, but under the hood, an array is just a block of memory where all elements sit `side by side`.
This figure below illustrates how array elements, like integers, are stored contiguously in memory. Each element occupies a specific block of memory, and for `int` types, these blocks are typically separated by `4 bytes`, allowing precise calculation of each element's address from the array's start. 
<div style="width: 70%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/array.svg" title="array" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Memory layout of a 4-element integer array with 4-byte spacing.
</div>
</div>

Because array elements are stored contiguously in memory, a pointer to the first element can be used to access any other element using pointer arithmetic. If `ptr` points to the start of the array, `ptr + i` moves the pointer `i` elements forward (not bytes — it accounts for the size of each element).
For example, to increment the third element (index 2):
```cpp
int arr[] = {1, 2, 3, 4};
int *ptr = arr;

ptr[2] = ptr[2] + 1;  // arr[2] becomes 4
```
Here, `ptr[2]` accesses the third element of the array, just like `arr[2]` would. This highlights the deep connection between arrays and pointers in C.

#### **2.4 2D Arrays and Memory Layout**
In C/C++, 2D arrays are stored in row-major order — meaning all elements of the first row come first in memory, followed by the second row, and so on. So even though we access elements using two indices (row and column), in memory it's just a flat, contiguous block. This is illustrated in the figure below.
<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/matrix.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
2D matrices, like this (3,4) example, are arranged row-major and stored contiguously in memory.
</div>
</div>

You can calculate the memory index of any element using:
> ```index = row * num_columns + col```

In the code below, `ptr[9]` accesses the same memory location as`matrix[2][1]`, because it is the 10th element in the row-major flattened memory layout (starting from index 0).
```cpp
int matrix[3][4] = {
    {5, 8, 9, 3},
    {11, 16, 1, 6},
    {6, 3, 8, 2}
};
int *ptr = &matrix[0][0];
int r = 2, c = 1;
int num_columns = 4;
int index = r * num_columns + c;
printf("%d\n", ptr[index]);  // Output: 3
```

##### **2.5 Stack vs Heap Memory**
This is often overlooked but important. In C/C++, small, fixed-size variables (like integers or small structs) are stored on the `stack` — a `fast`, temporary memory area that `automatically manages variable lifetime`.

In contrast, `dynamically allocated` memory (such as arrays created with `malloc` or `new`) lives on the `heap`, which is larger but slower and must be `manually managed` (allocated and freed).

```c
int x = 10;                 // Stored on the stack
int* arr = (int*)malloc(10 * sizeof(int));  // Allocated on the heap
```

We have now laid the essential groundwork of concepts—pointers, functions, memory layout, stack, and heap—on which CUDA programming is built.
Understanding these basics will make your journey into parallel programming much smoother.

#### **3. Your First CUDA Kernel : Hello World!**

Now that we have a solid understanding of the foundational concepts, let’s dive into writing our very first CUDA kernel. The goal here isn't complex computation, but to bridge the gap between CPU-style sequential thinking and GPU-style parallel execution, and to see your GPU actually *do something*.

A good way to learn something new, is to begin from something you already know and then connect the dots. Let us first look at a simple `Hello World` program in C++.
```cpp
#include <iostream>
int main(){
    std::cout << "Hello World!" << std::endl;
}
```
To execute this program on your machine, follow the following steps :

0. Clone the [repository](https://github.com/sanket-pixel/blog_code) that stores the code associated with my blogs :
```bash
git clone https://github.com/sanket-pixel/blog_code
cd blog_code
```

1. Navigate to the directory `8_that_first_cuda_blog/1_hello_world`
```bash
cd 8_that_first_cuda_blog/1_hello_world
 ```

 2. Compile the program using the following command 
 ```bash
g++ hello_world_cpu.cpp -o hello_world_cpu
 ```

 3. This will create an executable hello_world in this directory. Execute it using 
 ```bash
./hello_world_cpu
 ```

This should print `Hello World from CPU!` in the terminal, as expected. Here, the g++ compiler translates your C++ code into instructions that the CPU can execute directly.  

Finally, let's write our first CUDA kernel that performs the same task — printing "Hello World" — but this time the message will come from the GPU.

```cpp
#include <iostream>
#include <cuda_runtime.h>  // This include statement allows us to use cuda library in our code

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

1. Navigate to the directory `8_that_first_cuda_blog/1_hello_world`
```bash
cd 8_that_first_cuda_blog/1_hello_world
 ```

 2. Compile the program using the following command 
 ```bash
nvcc hello_world_gpu.cu -o hello_world_gpu
 ```

 3. This will create an executable hello_world in this directory. Execute it using 
 ```bash
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

Let us now extend our single thread CUDA Hello World, to run it with 8 threads. We would like the GPU to repeat this same `Hello World from GPU` operation 8 times. Just one small change in our original code will make this happen.


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


#### **4. Thread Organization in CUDA**

In the previous section, we briefly saw this line , ```gpu_hello_world<<<1,8>>>();```. We used it without explaining what ```<<<1,8>>>``` means. To truly understand CUDA programming, it’s important to unpack this syntax. This leads us to understanding how threads are organized in CUDA. 

##### **4.1 Heirarchy in CUDA**
When you launch a CUDA kernel, you typically launch many threads, not just one. These threads are organized in a hierarchical structure as shown in the figure below.

<div style="width: 70%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/heir.svg" title="heirarchy" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
CUDA hierarchy: A grid contains blocks, and each block contains threads.
</div>
</div>

Each kernel launch, creates one *Grid*, which contains many *Blocks*, which in turn contains many *Threads*.   
To understand how the threads, blocks and grids are organized, let us use the analogy of a building with *3 floors*, and with *4 apartments* on each floor. Thus we have 12 apartments in total.   
>Each *thread* is like an *apartment* (flat) on the floor.  
Each *block* is like a *floor* in the building.  
The *grid* is the entire *building*.  

This maps to `<<<3, 4>>>` in CUDA, which means launching 3 blocks, each containing 4 threads — totaling 12 threads. The two parameters inside the kernel launch syntax `<<<gridDim, blockDim>>>` represent the number of blocks in the grid (`gridDim`) and the number of threads in each block(`blockDim`).


##### **4.2 Finding global index of a thread**
In CUDA, thousands of threads may run in parallel, and each thread typically processes a different portion of data—like one element in an array. To do this correctly, each thread must know *exactly which piece of data it's responsible for*. That’s where the *global thread index* comes in: it gives every thread a unique ID across the entire grid so it can access the correct memory location.


<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/building.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
2D matrices, like this (3,4) example, are arranged row-major and stored contiguously in memory.
</div>
</div>

Extending our analogy as shown above, if every flat needs a unique global ID across the entire building, we do this:  
I’m on floor `floorIdx`, in flat `flatIdx`. Multiply the floor number by the number of flats per floor (`floorDim`) and add the flat index `flatIdx`.

```cpp
int global_flat_id = floorIdx * floorDim + flatIdx;
```

For example : 
```
Flat 2 on Floor 1: 1 * 4 + 2 = 6
Flat 3 on Floor 2: 2 * 4 + 3 = 11
```
In the figure above, `GID` refers to the `global_flat_id` . As shown on the left, every flat has a unique ID ranging from 0 to 11.  


Along similar lines, to get the *global thread index*, we can use the `blockIdx`, `blockDim` and `threadIdx`. CUDA gives us built-in variables 
to retrieve this information:
> `blockIdx.x` → which block (floor) you’re in  
`threadIdx.x` → which thread (room) inside the block  
`blockDim.x` → how many threads per block (rooms per floor)

Using these, every thread can compute its global ID using:
```cpp
int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
```
This index lets the thread determine exactly which data element to work on.  
Let’s look at a simple example where each thread doubles one element in an array:
```cpp
__global__ void double_elements(int* arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    arr[idx] = arr[idx] * 2;
}
```
In this kernel, every thread calculates its global index and uses it to access the correct element in `arr`. Thanks to this indexing, all threads operate independently, and no two threads overwrite each other's work.


##### **4.3 A Hands-on Example: Squaring Numbers in Parallel**
Let us now walk through a concrete example to reinforce what we’ve learned about thread organization in CUDA. This example not only demonstrates how threads are structured using blocks and grids but also highlights some important subtleties that apply to most CUDA kernels you will write.  

We’ll square numbers from 0 to 9 in parallel. Each thread computes its global ID and prints the square of its assigned number.
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 4
#define DATA_SIZE 10

__global__ void print_square() {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < DATA_SIZE) {
    printf("Thread %d: %d squared = %d\n", id, id, id * id);
  }
}

int main() {
  // compute how many blocks we need
  int blocks = (DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  print_square<<<blocks, THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
}
```
How to run it:
1. Navigate to the directory:
```bash
cd 8_that_first_cuda_blog/2_squared_numbers
 ```

 2. Compile the program:
 ```bash
nvcc squared_numbers.cu -o squared_numbers
 ```

 3. Run the executable:
 ```bash
./squared_numbers
 ```

 4. You’ll see output like:
 ```
Thread 0: 0 squared = 0
Thread 1: 1 squared = 1
Thread 2: 2 squared = 4
// and so on...
 ```

This diagram below visually explains how CUDA computes each thread’s global ID using `blockIdx.x`, `threadIdx.x`, and `blockDim.x`.  It also shows which threads execute the computation based on the condition `if (id < DATA_SIZE)`. 

<div style="width: 80%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/square.svg" title="square" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
2D matrices, like this (3,4) example, are arranged row-major and stored contiguously in memory.
</div>
</div>


Let's break down the key ideas:
- We define two constants at the top:
  - `THREADS_PER_BLOCK = 4` → Each block contains 4 threads.
  - `DATA_SIZE = 10` → We want to square numbers from 0 to 9.  
<br>
- In `main()`, we compute how many blocks we need. As shown in the figure above, `int blocks =3`:
```cpp
int blocks = (DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
```
- We launch the kernel:
```cpp
print_square<<<blocks, THREADS_PER_BLOCK>>>();
```
- Inside the kernel, each thread calculates its global ID:
```cpp
unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
```
- The check `if (id < DATA_SIZE)` ensures that no thread attempts to process an index beyond the data size — an essential safeguard when the total number of threads exceeds the amount of data.


##### **4.4 Thread and Block IDs in 2D and 3D**
So far, we’ve been using variables like `threadIdx.x`, `blockIdx.x`, and `blockDim.x` in our CUDA kernels — but we haven’t really explored what the `.x` means, or what happens when we go beyond 1D.

Understanding how threads and blocks are structured in *multiple dimensions (2D and 3D)* is essential because many real-world problems are naturally multi-dimensional — like image processing (2D pixels), volumetric data (3D grids), or matrix operations. CUDA’s thread hierarchy lets us mirror this structure directly, so we can write cleaner, more intuitive code.

In CUDA, both threads and blocks can be organized in 1D, 2D, or 3D layouts. That means:
- Each *block* can have threads arranged like a line (1D), a grid (2D), or a cube (3D).
- Similarly, the *grid of blocks* itself can follow any of these layouts.

Each thread and block has 3 coordinate components:
- `threadIdx.x`, `threadIdx.y`, `threadIdx.z` tell you the thread’s position **within its block**.
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z` tell you the block’s position **within the grid**
- `blockDim.x`, `blockDim.y`, `blockDim.z` tell you how many threads are there **per block** in each direction

This gives every thread a unique multi-dimensional ID — but to index into flat memory like an array, we typically compute a flattened 1D global index from these coordinates.

**2D grid and blocks**  
Let's say each block is a 2D grid of threads and the entire grid has a 2D arrangement of blocks. Then, we compute a *flattened global thread ID* like this:
```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```
You now have coordinates `(x, y)` representing the thread’s unique position in the global 2D grid. If you want to flatten this into a 1D index (e.g. for array access), you can do:
```cpp
int idx = y * total_width + x;  // where total_width is the width of the full grid
```

**3D grid and blocks**  
In case the grid has blocks in 3D, and the blocks have threads in 3D,
```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```
And to get a linear index:
```cpp
int idx = z * (height * width) + y * width + x;
```

These computations let each thread know *exactly what data to work on*, even in multi-dimensional problems. CUDA doesn't care if you use 1D, 2D, or 3D — it just provides the structure so you can map the problem domain naturally and write code that’s easier to reason about.
>TL;DR: Think of .x, .y, and .z as the coordinates in a virtual 3D thread universe. You use them to uniquely identify and assign work to each thread, especially in problems where your data naturally lives in 2D or 3D.

Let's Understand This with a Concrete 2D Example  

```cpp
#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 10      // width of simulated 2D data (columns)
#define HEIGHT 5      // height of simulated 2D data (rows)

#define THREADS_X 4   // threads per block in X
#define THREADS_Y 2   // threads per block in Y

__global__ void print_2d_coordinates() {
  // Get global 2D index
  int global_x = blockIdx.x * blockDim.x +  threadIdx.x;
  int global_y = blockIdx.y * blockDim.y + threadIdx.y;

  // Compute linear ID (row-major order)
  int global_id = global_y * WIDTH + global_x;

  if (global_x < WIDTH && global_y < HEIGHT) {
    printf("Block (%d,%d) Thread (%d,%d) → Global ID: %2d → Pixel (%d,%d)\n",
           blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, global_id, global_y, global_x);
  }
}

int main() {
  dim3 threads_per_block(THREADS_X, THREADS_Y);

  int blocks_x = (WIDTH + THREADS_X - 1) / THREADS_X;
  int blocks_y = (HEIGHT + THREADS_Y - 1) / THREADS_Y;
  dim3 num_blocks(blocks_x, blocks_y);

  print_2d_coordinates<<<num_blocks, threads_per_block>>>();
  cudaDeviceSynchronize();
}
```

To execute this program on your machine, follow the following steps :

1. Navigate to the directory *8_that_first_cuda_blog/3_print_2d_coordinates*
```bash
cd 8_that_first_cuda_blog/3_print_2d_coordinates
 ```

 2. Compile the program using the following command 
 ```bash
nvcc print_2d_coordinates.cu -o print_2d_coordinates
 ```

 3. This will create an executable `print_2d_coordinates` in this directory. Execute it using 
 ```bash
./print_2d_coordinates
 ```

 4. The output will start with something like :
 ```
Block (2,2) Thread (0,0) → Global ID: 48 → Pixel (4,8)
Block (2,2) Thread (0,1) → Global ID: 49 → Pixel (4,9)
Block (2,1) Thread (0,0) → Global ID: 44 → Pixel (4,4)
Block (2,1) Thread (0,1) → Global ID: 45 → Pixel (4,5)
Block (2,1) Thread (0,2) → Global ID: 46 → Pixel (4,6)
Block (2,1) Thread (0,3) → Global ID: 47 → Pixel (4,7)
// and so on..
 ```


In this code :
- We're simulating a **2D image** of size `10 x 5` (10 columns × 5 rows).
- Each block is configured to launch 4 x 2 threads. ( blockDim.x=4, blockDim.y=2)\
- Based on the data dimensions and thread layout, we calculate how many blocks we need in X and Y directions:
```cpp
int blocks_x = (WIDTH + THREADS_X - 1) / THREADS_X;
int blocks_y = (HEIGHT + THREADS_Y - 1) / THREADS_Y;
```
- Inside the kernel, each thread computes:
  - Its global X and Y position using its thread and block indices
  - A linear global ID, assuming row-major layout  

  ```cpp
  // Get global 2D index
  int global_x = blockIdx.x * blockDim.x +  threadIdx.x;
  int global_y = blockIdx.y * blockDim.y + threadIdx.y;

  // Compute linear ID (row-major order)
  int global_id = global_y * WIDTH + global_x;
  ```
- The `if` condition:

  ```cpp
  if (global_x < WIDTH && global_y < HEIGHT)
  ```
ensures that only threads within bounds print output. Without this, some threads (especially in the last block row/column) might run out-of-bounds and access invalid pixels.
- In the earlier 1D example, we used plain integers like `<<<blocks, threads>>>`. In this 2D case, we use `dim3` to pass 2D configurations — a CUDA struct that lets us naturally map threads to 1D, 2D, or 3D data layouts.

<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/2d.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Computing the global thread ID from the block index and thread index in a 2D grid configuration.
</div>
</div>

As shown above, for an image of `10x5` with and `blockDim` of `(4,2)`, the `gridDim` is `(3,3)`. To compute the global thread ID of 
the thread marked in red :

  ```
blockIdx = (1, 0) → 2nd block in X, 1st in Y ( highlited in yellow )
threadIdx = (1, 2) → row 1, column 2 within the block ( highlited in red)
global_x ​= blockIdx.x × blockDim.x + threadIdx.x
        = 1 x 4 + 2 = 6

global_y ​= blockIdx.y × blockDim.y + threadIdx.y
        = 0 x 2 + 1 = 1

global_id = global_y x WIDTH + global_x
          = 1 x 10 + 6 = 16

  ```

That wraps up our section on thread organization — we now understand how threads and blocks are structured and how they map to data. Whether 1D or 2D, it's all about aligning the thread layout to your problem.

Next, we’ll explore how memory works in CUDA — including how to allocate memory on the GPU and transfer data between the CPU and GPU.

#### **5. Managing Data: From CPU to GPU and Back**
In this section, we’ll understand how data moves between the CPU and GPU, and why explicit memory management is crucial in CUDA programming.

Here, we only see the essentials, but you can check out my previous blog post [Down the CUDA Memory Lane](/blog/2024/down-the-cudamemory-lane) for a deep dive into CUDA Memory.

Before diving deeper, it’s important to understand a key idea —**the CPU (host) and GPU (device) have separate memory spaces**. They do not share memory by default and cannot directly access each other’s data. If you want the GPU to work on data from the CPU (or send results back), **you must explicitly transfer that data between the two.**

There are a few subtle but essential things to keep in mind:
- Simply defining an array on the CPU doesn’t make it visible to the GPU.
- The GPU cannot allocate or manage host memory directly.
- Data must be copied from host to device before the kernel runs, and back afterward if needed.
- Memory transfers are relatively expensive compared to computation, so minimizing them is often important for performance.

In this section, we’ll **briefly summarize how to allocate memory using `cudaMalloc` and copy data with `cudaMemcpy`**, just enough to get us started with real examples.

As always, lets first directly look at a working example and then understand these memory management thought that.

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

In this example, we prepare a list of 10 numbers on the CPU, send it to the GPU, bring it back, and check that everything transferred correctly.
To achieve this, we perform the following operations :
In this example, we will perform the following operations:

- Allocate memory on the CPU for an integer array of size 10   
```cpp 
int *h_array = (int*) malloc(size);
```
`malloc`, short for “memory allocation,” is a standard C/C++ function that reserves a block of memory on the CPU. 
- Allocate memory on the GPU for an integer array of size 10
```cpp
cudaError_t err = cudaMalloc((void**)&d_array, size);
```
`cudaMalloc` is the CUDA equivalent of `malloc`, used to allocate memory on the GPU.Even though d_array is created in CPU code, it doesn’t store a regular number — it stores the address of memory that lives on the GPU. It’s like a note on the CPU that says, *“Hey, the actual data is over there on the GPU.”* The function takes a pointer to the pointer (&d_array) and the size in bytes.<br> <br>
Since `cudaMalloc` expects a `void**`, we pass the address of the pointer (`&d_array`) so it can fill it with the location of the allocated GPU memory. Think of it like this: we’re giving CUDA a place to *write down the GPU address*, and after the call, `d_array` will hold the actual memory location on the GPU. It’s a bit of a tongue twister — passing the address of an address — but that’s how the pointer gets set correctly.
<div style="width: 70%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/cudamalloc.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Computing the global thread ID from the block index and thread index in a 2D grid configuration.
</div>
</div>

- Initialize the CPU array with squares of integers from 1 to 10
```cpp
  for (int i = 0; i < 10; ++i) {
        h_array[i] = (i + 1) * (i + 1); // Squares of integers from 1 to 10
    }
```


- Copy the data from the CPU to the GPU
```cpp
err = cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
```
`cudaMemcpy` is used to transfer data between the CPU and GPU. Here, we’re copying data from the CPU array `h_array` to the GPU array `d_array`. The last argument, `cudaMemcpyHostToDevice`, tells CUDA the direction of the copy. This step is crucial — the GPU can't access CPU memory directly, so we have to send it over manually.
- Copy the data back from the GPU to the CPU 
```cpp
err = cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
```  
This line copies the results from GPU memory (`d_array`) back to CPU memory (`h_array`). The direction flag `cudaMemcpyDeviceToHost` tells CUDA we’re transferring data from the GPU to the CPU.
<div style="width: 70%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/cudamemcpy.png" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Computing the global thread ID from the block index and thread index in a 2D grid configuration.
</div>
</div>

With this, we’ve covered the basics of how to manage memory between the CPU and GPU — a crucial foundation for any CUDA program.


#### **6. Your First Real CUDA Example: Grayscale Conversion**
We’ve now covered key CUDA concepts like thread organization, memory management, and kernel launches, and written several simple toy kernels to make them stick. It’s time to take off the training wheels and write a full CUDA kernel to solve a real-world problem.

In this next section, we’ll convert a color image to grayscale — not one pixel at a time like we would on the CPU, but all at once by leveraging CUDA’s parallel threads. It’s a practical use case that brings everything we’ve learned together. Let us first look at the code and run it locally to convert a sample color image to grayscale.

<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/gray.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Computing the global thread ID from the block index and thread index in a 2D grid configuration.
</div>
</div>


##### **6.1 Understanding Image Data: RGB, Grayscale, and Memory Layout**
Before we dive into writing a CUDA kernel for image processing, we need to understand how image data is actually stored in memory. This section provides a foundational overview for readers who are comfortable with programming but new to image manipulation.

Most color images use the RGB format, where each pixel consists of three values: *red intensity, green intensity, and blue intensity*. Each of these values typically occupies `1 byte (0–255)`, meaning a single RGB pixel takes up 3 bytes in memory. In the image below, the pixels marked A, B, and C each represent such RGB triplets, with their respective red, green, and blue components visualized. This structure forms the foundation of how color is encoded and stored in digital images — each pixel is just a tiny combination of three color intensities.

<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/rgb.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Each pixel (A, B, C) stores color using three values — red, green, and blue — forming an RGB triplet.
</div>
</div>

If the image has a width of `W` and a height of `H`, then the RGB image is stored in memory as a 1D array of size `H x W x 3`. The storage is typically *row-major*, meaning we store pixels row by row. For example, the first row’s pixels come first, then the second row’s, and so on.

For a `3×2` image`(3 columns, 2 rows)`, the RGB memory layout looks like this:

<div style="width: 70%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/pixels.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
</div>

This is flattened into a 1D array as:
```
[R00, G00, B00, R01, G01, B01, R02, G02, B02, R10, G10, B10, R11, G11, B11, R12, G12, B12]
```
To locate the RGB triplet for a pixel at `(row, col)`:
```cpp
int index = (row * width + col) * 3;
unsigned char r = input[index];
unsigned char g = input[index + 1];
unsigned char b = input[index + 2];
```

A grayscale image stores only one intensity value per pixel — no color, just brightness. This simplifies both the memory and computation.
For the same `3×2 image`, a grayscale layout would be:
<div style="width: 60%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/graypixel.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
</div>
Flattened memory:
```
[P00, P01, P02, P10, P11, P12]
```
Only one byte per pixel is stored, and the total memory size is `HxW` bytes. To access the grayscale value for a pixel at `(row, col)`:

```cpp
int index = row * width + col;
unsigned char intensity = output[index];
```
The grayscale intensity `P` for an RGB pixel is typically calculated using the following weighted average:
```python
L = 0.299 * R + 0.587 * G + 0.114 * B
```
This formula accounts for human visual sensitivity to different colors and is widely used in image processing.  
This basic understanding of how pixel data is organized in memory sets the stage for the CUDA implementation in the next section, where each GPU thread will process one pixel at a time — reading its RGB triplet, converting it to grayscale, and writing the result into the output buffer.

##### **6.2 Converting RGB to Grayscale in CUDA**
In this next section, we’ll convert a color image to grayscale — not one pixel at a time like we would on the CPU, but all at once by leveraging CUDA’s parallel threads. It’s a practical use case that brings everything we’ve learned together. Let us first look at the code and run it locally to convert a sample color image to grayscale.

```cpp
#include <cuda.h>
#include <opencv2/opencv.hpp>
#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32
using namespace std;

__global__ void convert_rgb_to_grayscale(float *dsample_image,
                                         float *dgrayscale_sample_image,
                                         int rows, int cols) {
  // compute the (x, y) coordinates of the thread in the image
  int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  int global_y = blockIdx.y * blockDim.y + threadIdx.y;

  // check that we are within image bounds
  if (global_x < cols && global_y < rows) {
    // flatten 2D coordinates into a 1D index
    int global_id = global_y * cols + global_x;

    // fetch the RGB values for the current pixel
    float r = dsample_image[3 * global_id];
    float g = dsample_image[3 * global_id + 1];
    float b = dsample_image[3 * global_id + 2];

    // compute grayscale using weighted sum (perceptual luminance)
    dgrayscale_sample_image[global_id] = 0.144 * r + 0.587 * g + 0.299 * b;
}


int main() {
  // read image from filepath
  string sample_image_path = "../sample.png";
  cv::Mat sample_image = cv::imread(sample_image_path);
  int width = sample_image.cols;
  int height = sample_image.rows;
  int channels = sample_image.channels();
  sample_image.convertTo(sample_image, CV_32F, 1.0 / 255.0);

  // allocate memory on GPU
  int sample_image_size_in_bytes = width * height * channels * sizeof(float);
  float *dsample_image;
  cudaMalloc(&dsample_image, sample_image_size_in_bytes);
  // allocate memory on GPU to store the grayscale image
  float *dgrayscale_sample_image;
  cudaMalloc(&dgrayscale_sample_image, width * height * sizeof(float));

  // copy image from CPU to GPU
  cudaMemcpy(dsample_image, sample_image.data, sample_image_size_in_bytes,
             cudaMemcpyHostToDevice);

  // compute number of blocks in x and y dimensions
  int number_of_blocks_x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
  int number_of_blocks_y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
  // define grid dimension and block dimension for kernel launch
  dim3 grid_dim(number_of_blocks_x, number_of_blocks_y, 1);
  dim3 block_dim(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  // launch the kernel
  convert_rgb_to_grayscale<<<grid_dim, block_dim>>>(
      dsample_image, dgrayscale_sample_image, height, width);


  // copy the grayscale image back from GPU to CPU
  cv::Mat himage_grayscale(height, width, CV_32FC1);
  float *himage_grayscale_data =
      reinterpret_cast<float *>(himage_grayscale.data);
  cudaMemcpy(himage_grayscale_data, dgrayscale_sample_image,
             width * height * sizeof(float), cudaMemcpyDeviceToHost);
  himage_grayscale.convertTo(himage_grayscale, CV_8U, 255.0);
  cv::imwrite("../grayscale_sample.png", himage_grayscale);
  return 0;
}
```

To execute this program on your machine, follow the following steps :

1. Navigate to the directory *8_that_first_cuda_blog/3_print_2d_coordinates*
```bash
cd 8_that_first_cuda_blog/4_grayscale_2d
 ```
2. Make a `build` directory and navigate into it
```bash
mkdir build && cd build
```
3. Generate the Makefile. Specify the appropriate CUDA path 
```bash
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
```
 4. Compile the program using the following command 
 ```bash
make
 ```

 5. This will create an executable `grayscale_2d` in the `build` directory. Execute it using 
 ```bash
./grayscale_2d
 ```

 6. The grayscale image will be stored in the *8_that_first_cuda_blog/4_grayscale_2d* folder as `grayscale_sample.png`


If all went well, you just wrote your first real and useful CUDA kernel. In essence, the code above does the following:
1. We loaded a color image using OpenCV and converted its pixel values to floats.
2. We allocated memory on the GPU for both the input image and the grayscale output.
3. We copied the image data from the CPU to the GPU.
4. We calculated how many blocks and threads we need to cover all pixels—each block handles a small tile of the image, and each thread processes one pixel.
5. The CUDA kernel ran in parallel, where each thread took one pixel and computed the grayscale value as a weighted sum of its R, G, B components.
6. Finally, we copied the grayscale result back to the CPU and saved it as an image file.

Let us look at the above code, one crucial part at a time.

- Firstly, we read the image using `OpenCV`, extract the `width`,`height` and `channels`. 
- By default, OpenCV loads the image in `int8` format. We convert it to `float32` for ease of processing. <br> 
```cpp
// read image from filepath
string sample_image_path = "../sample.png";
cv::Mat sample_image = cv::imread(sample_image_path); // read the image
int width = sample_image.cols; // extract height, width, channels
int height = sample_image.rows;
int channels = sample_image.channels();
sample_image.convertTo(sample_image, CV_32F, 1.0 / 255.0); // convert the image from int8 to float32
```
The details of OpenCV and its interfaces are beyond the scope of this blog post. 
For our purposes, it is enough to understand that the code above simply loads the image into CPU memory in a row-major, contiguous format.
- We then calculate how much memory the image will occupy in bytes `(width × height × channels × sizeof(float))`.
- Then we allocate that much space on the GPU using `cudaMalloc`.
- We copy the image data from the CPU (`sample_image.data`) to the allocated GPU memory (`dsample_image`) using cudaMemcpy.
This ensures that the image is now available on the GPU for parallel processing.
```cpp
int sample_image_size_in_bytes = width * height * channels * sizeof(float);
float *dsample_image;
cudaMalloc(&dsample_image, sample_image_size_in_bytes);
cudaMemcpy(dsample_image, sample_image.data, sample_image_size_in_bytes,
          cudaMemcpyHostToDevice);
```
- `dsample_image` is the pointer to the input RGB image, and `dgrayscale_sample_image` is the pointer to the output grayscale image.
We pass pointers because CUDA kernels operate on data already present on the GPU — they don’t copy data themselves.
- We also pass the image dimensions (`rows` and `cols`) so that each thread can figure out which pixel it is responsible for.
```cpp
__global__ void convert_rgb_to_grayscale(float *dsample_image,
                                         float *dgrayscale_sample_image,
                                         int rows, int cols)
```
- Each GPU thread is assigned a unique (x, y) coordinate using its block and thread indices.
- It checks whether this coordinate lies within the bounds of the image to avoid illegal memory access.
- Using the 2D coordinate, it calculates a flat 1D index to access the RGB values of that pixel.
- Finally, it computes the grayscale value using a weighted sum and writes it to the output array on the GPU.
```cpp
int global_x = blockIdx.x * blockDim.x + threadIdx.x;
int global_y = blockIdx.y * blockDim.y + threadIdx.y;
if (global_x < cols && global_y < rows) {
  int global_id = global_y * cols + global_x;
  float r = dsample_image[3 * global_id];
  float g = dsample_image[3 * global_id + 1];
  float b = dsample_image[3 * global_id + 2];
  dgrayscale_sample_image[global_id] = 0.144 * r + 0.587 * g + 0.299 * b;
}
```

- We calculate how many blocks are needed in the X and Y directions to cover the entire image, rounding up to handle any leftover pixels.
- Each block contains a fixed number of threads defined by `BLOCKSIZE_X` and `BLOCKSIZE_Y`.
- These values are used to create the grid and block dimensions (`grid_dim` and `block_dim`), which tell CUDA how to organize threads for parallel execution.
- Finally, we launch the kernel with this configuration, passing the GPU pointers and image dimensions so each thread can process its assigned pixel.
```cpp
// compute number of blocks in x and y dimensions
int number_of_blocks_x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
int number_of_blocks_y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
// define grid dimension and block dimension for kernel launch
dim3 grid_dim(number_of_blocks_x, number_of_blocks_y, 1);
dim3 block_dim(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
// launch the kernel
convert_rgb_to_grayscale<<<grid_dim, block_dim>>>(
    dsample_image, dgrayscale_sample_image, height, width);
```
- We create an empty OpenCV matrix `himage_grayscale` on the CPU to hold the grayscale image data and get a raw pointer to its data.
- Then, we copy the grayscale image from GPU memory back to the CPU, convert it to an 8-bit format, and save it as a PNG file using OpenCV’s `imwrite`.
```cpp
// copy the grayscale image back from GPU to CPU
cv::Mat himage_grayscale(height, width, CV_32FC1);
float *himage_grayscale_data =
    reinterpret_cast<float *>(himage_grayscale.data);
cudaMemcpy(himage_grayscale_data, dgrayscale_sample_image,
            width * height * sizeof(float), cudaMemcpyDeviceToHost);
himage_grayscale.convertTo(himage_grayscale, CV_8U, 255.0);
cv::imwrite("../grayscale_sample.png", himage_grayscale);
  ```









<!-- One important concept regarding host-device communication is that the **host does not wait for the kernel execution to finish**, and moves on with the next instruction. This execution approach is known as **asynchronous**. In particular, the `host` and `device` executes independently and simulatenously. When a command is like kernel launch is issued by the host, it does not wait for the command to complete on the `device`, but simply moves on to the next instruction, while the `device` handles the requested operation in parallel.

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

Note, that since we removed `cudaDeviceSynchronize();`, the host launches the `gpu_hello_world` kernel and moves on to the next instruction. The exection of the host code finishes, even before the `device` completes, hence it does not print `Hello World from GPU!` onto the output buffer. This simple example highlights the separation between CPU and GPU execution — each runs independently unless explicitly synchronized. -->