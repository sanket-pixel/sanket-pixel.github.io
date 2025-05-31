---
layout: post
title: That First CUDA Blog I Needed :Part 3
date: 2025-05-31 08:53:00-0400
description: Solving a Real World Problem with CUDA
thumbnail : /assets/img/blog/blog_8/uni.jpeg
categories: cuda
tag : [nvidia, cuda]
giscus_comments: false
related_posts: true
---

In the previous part of this blog, [Part 2: Building Blocks of Parallelism](/blog/2025/that-first-cuda-blog-2), we explored how CUDA organizes threads into blocks and grids, and how memory is managed between the CPU and GPU. That gave us the tools. Now, in Part 3, we bring it all together in a real CUDA project — processing an image on the GPU, handling real-world memory issues, and learning from common beginner mistakes.

<br>
<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/uni.jpeg" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   Part 3 : Graduating in CUDA 
</div>
</div>


##### [Part 3: A Real-World CUDA Project](/blog/2025/that-first-cuda-blog-3) 
[6. Your First Real CUDA Example: Grayscale Conversion](/blog/2025/that-first-cuda-blog-3#6-your-first-real-cuda-example-grayscale-conversion)  
[7. Common Pitfalls When Getting Started](/blog/2025/that-first-cuda-blog-3#7-common-pitfalls-when-getting-started)  
[8. That’s a Wrap — Now You’re CUDA-Capable](/blog/2025/that-first-cuda-blog-3#8-thats-a-wrap-now-youre-cuda-capable)

 > All the code related to this blog series, accompanying each step of your CUDA learning journey, can be found on GitHub at: [https://github.com/sanket-pixel/blog_code/tree/main/8_that_first_cuda_blog](https://github.com/sanket-pixel/blog_code/tree/main/8_that_first_cuda_blog).

### **6. Your First Real CUDA Example: Grayscale Conversion**
We’ve now covered key CUDA concepts like thread organization, memory management, and kernel launches, and written several simple toy kernels to make them stick. It’s time to take off the training wheels and write a full CUDA kernel to solve a real-world problem.

In this next section, we’ll convert a color image to grayscale — not one pixel at a time like we would on the CPU, but all at once by leveraging CUDA’s parallel threads. It’s a practical use case that brings everything we’ve learned together. Let us first look at the code and run it locally to convert a sample color image to grayscale.

<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/gray.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Converting colored image to grayscale using CUDA
</div>
</div>


##### **6.1 Understanding Image Data: RGB, Grayscale, and Memory Layout**
Before we dive into writing a CUDA kernel for image processing, we need to understand how image data is actually stored in memory. This section provides a foundational overview for readers who are comfortable with programming but new to image manipulation.

Most color images use the RGB format, where each pixel consists of three values: **red intensity, green intensity, and blue intensity**. Each of these values typically occupies `1 byte (0–255)`, meaning a single RGB pixel takes up 3 bytes in memory. In the image below, the pixels marked A, B, and C each represent such RGB triplets, with their respective red, green, and blue components visualized. This structure forms the foundation of how color is encoded and stored in digital images — each pixel is just a tiny combination of three color intensities.

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

1. Navigate to the directory *8_that_first_cuda_blog/4_grayscale_2d*
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

<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/flow.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
</div>

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

And with that, we complete converting an colored image to grayscale using CUDA as shown below. Each pixel is processed by one thread, in parallel on the GPU.

<div style="width: 90%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_8/gray.svg" title="matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Left : Colored input image Right : Grayscale output image
</div>
</div>


### **7. Common Pitfalls When Getting Started**
Even with the basics in place, beginners often run into a few predictable issues when working with CUDA for the first time. Here are some of the most common ones to watch out for:

**Threads are not sequential**
Just because you launch threads with increasing indices (threadIdx.x = 0, 1, 2...) doesn’t mean they run in that order. Threads execute in parallel, and their actual scheduling is unpredictable. Don’t write code that assumes a specific order of execution.

**Thread limit per block**
The maximum number of threads per block is typically 1024. If you accidentally set a higher block dimension (say, blockDim.x = 2048), the kernel will silently fail or produce garbage results—often all zeros. Always check that your block configuration respects this hardware limit.

**Not syncing when CPU depends on GPU**
CUDA kernel launches are asynchronous. If your CPU code depends on the GPU result right after a kernel call, you must call cudaDeviceSynchronize() to wait for GPU completion before using the data.

**Mixing up grid and block indices**
It's easy to confuse blockIdx and threadIdx, or miscalculate global thread IDs. Always double-check your formulas when computing pixel indices or array offsets.

**Forgetting to check memory copies**
Many issues arise from not copying data to or from the GPU at the right time. Use cudaMemcpy() carefully and verify its direction (cudaMemcpyHostToDevice vs. DeviceToHost).

**Uninitialized or out-of-bounds memory**
Accessing memory outside the range of your arrays won’t throw an error—it just causes silent corruption or crashes. Always make sure your thread doesn't go beyond valid bounds, especially when using 2D grids or blocks.

**Kernel launch succeeds but does nothing**
A kernel that silently does nothing can happen due to:
- Launching with zero threads.
- All threads exiting early due to an if condition.
- Threads writing to out-of-bounds memory.
- Use cudaGetLastError() to check for launch issues.

### **8. That’s a Wrap: Now You’re CUDA-Capable**
If you made it this far, you’ve already done more than most who skim CUDA docs and bounce off. The goal wasn’t to turn you into a GPU performance wizard overnight — it was to flip the mental switch. To make CUDA feel a little less alien.

We didn’t chase shared memory or benchmark numbers. We stayed grounded — loading an image, writing a basic kernel, seeing how threads work, and gently peeling back the hardware layers. And that’s enough. Because *truly understanding one kernel end to end* teaches you more than stitching together code from Stack Overflow.

The truth is, you’ll forget the exact syntax. You might mix up `threadIdx` and `blockIdx` next week. But what stays is the *shape* of the model: blocks, threads, grids — and how they map to real computation. That shape is what lets you grow later.

You don’t need to master every corner of the GPU before you're allowed to use it. Write something small. Let it run. Watch it scale. That’s already winning.

So if you’re walking away from this thinking, “Wait, I can actually write CUDA now,” — that’s exactly the feeling I wanted this blog to give you.

Onward.
