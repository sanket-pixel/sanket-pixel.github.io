---
layout: post
title: "CUDA-ify Your Point Clouds"
date: 2023-07-12 15:53:00-0400
description: Voxelization using CUDA programming
categories: cuda
tag : [nvidia, cuda]
giscus_comments: false
related_posts: true
---



#### Experience the exhilarating world of CUDA programming as we revolutionize point cloud processing in computer vision and robotics. Voxelization, the process of converting 3D points into discrete voxels, has faced challenges with traditional CPU-based methods, limiting groundbreaking innovations. But fear not, as we harness the immense power of parallelization for a monumental leap in processing speed! Dive into CUDA's awe-inspiring realm, where each point gets its own thread, enabling lightning-fast voxelization and opening the doors to real-time applications. Join us on this thrilling ride and witness the magic of CUDA as we rewrite the future of point cloud processing. Let's embrace the sheer power of CUDA together and change the game!


<!-- ### Notes on how the logic works
1. The point clouds are loaded directly from file and copied to device.
2. We make the hash table for the given input points.
    - The hash table will store all unique_voxels as keys and the voxel_id for each unique voxel as ID.This means that we only store information about voxels that      actually have a point in them. The voxels are flattened out to calculate offset and the hashed value of this flattened out voxel acts as key. The value is the 
    index of the voxel in this sequence of unique voxels.
    - Each point is processed on a seperate thread. First we compute the voxel_x, voxel_y, voxel_z for this point using general point to voxel clamping.
    - Then, we insert the voxel_offset ( flattened out voxel id ) as key, current total unique voxels as value in a hash table. 
        - The hash table stores all keys ( voxel offsets ) and value (voxel_id) serially. So if there are 100 unique voxels, the hash table will be of size 200,
          and store 100 keys first, followed by 100 values. 
        - To that end, we first calculate hash value of the key. The modulus function is applied on this hash value ( of hash_size/2) to find the slot where this 
          key would fit in the hash table.
        - So if the hash value of the key is 126, then the value after modulus will become 126%100 = 26. This means the key will be stored at index 26 and value will 
          be stored at (26+100) = 126.
        - We apply Compare and Swap function on this key. But there are three possible conditions while we apply AtomicCAS function on the key at the given slot. 
          It returns the current key in the slot. 3 possibilities :
            1. If empty, that means the insertion was successful. We then insert corresponding value
               which is the unique voxel index.
            2. If key matches current key, it means this voxel offset is already in hash table.
            3. If key is another key, that is a collision, which means a different key with same hash value exists in table.
               Apply linear probing to solve this problem. Check in the next slot.
3. Then we apply Voxelization using the hash map we created. Again every point is processed on separate thread. The goal is to create an array called 
   voxel_temp of size (max_voxels * max_points_per_voxel * num_features) where all point features for all voxels are stored serially where features from points
   from the same voxel are together in memory. 
   - First we compute voxel_x, voxel_y and voxel_z for the current point using clamping. 
   - Then we compute the voxel_offset by flattening out the voxels from 3D to 2D and finding the linear index. 
   - Then we pass the voxel_offset as key to lookup in the hash table. This gives us the corresponding value for this voxel_offset key,
     which is the voxel_id. The voxel_id is the index of the voxel in unique voxels.
   - Now we update the number of points in the current voxel by 1.
   - Then, we calculate the dst_offset using voxel_id, max_points_per_voxel, feature_num, points_in_voxel.
   - We compute the src_offset using point_idx and feature_num from points array.
   - We now copy the current points feature from points array to voxel_temp array. 
   - voxel_temp array stores all points features of all voxels,
    in a serialized fashion in order of the voxel_id. For for example, for first voxel, ( if max points per voxel = 3),
    voxel_temp stores [x11,y11,z11,i11,t11,x12,y12,z12,i12,t12,x13,y13,z13,i13,t13]. where i featureij indicates voxelid and j indiciates pointid i
    in that voxel and feature is x,y,z, etc.
   - We also store the voxel indices (b,x,y,z) in voxel_indices array using voxel_id.

4. Now we begin the feature extraction kernel. We have all point features for all points stored serially in order of voxel_ids in voxel_temp. Now all that remains
   is to take average of all these features (x,y,z,i,t) for all points for each voxel and store them in a voxel_features array.
   - The feature extraction process happens on the voxel level. So now, we use one thread for every voxel. 
   - First we upgrade the num_points_per_voxel array for this voxel.
   - We will be updating the feature values of the first point of every voxel with the average of all features of all points in that voxel.
   - To that end, we first compute offset for this voxel in the array which will be used to access this voxels first point in voxel_temp.
   - Now we iterate over every feature and then over every point in this voxel, starting from second point ( because first is to be updated).
   - Then we sum up all points features in this voxel and take average by dividing by total points.
   - Finally the voxel_features array is updated using these averaged out features. Now we have features for every voxel in coninous memory in voxel_features array. -->

<!-- <br>

## 1. Intuition 
In this section, we will embark on an intuitive exploration of the point cloud voxelization process using CUDA. We'll begin by understanding the creation of a hash table, which acts as the foundation for this accelerated voxelization. Next, we'll dive into the fascinating voxelization process itself, where each point gets its own dedicated thread, enabling rapid parallel computation. Lastly, we'll unravel the final feature extraction,where each voxel gets it's own thread to bring the power of voxelized data to life. Brace yourself for an exhilarating journey through the concepts that revolutionize point cloud processing. Let's get started!

### 1.1 The Key Steps

## 2. Inside the Code
In this section, we will take a comprehensive look at the CUDA-powered voxelization implementation. We'll dissect the code, step by step, to understand the magic behind its blazing-fast performance. To begin our exploration, let's first examine the folder structure that forms the foundation of this efficient voxelization engine. Understanding the organization of the code will serve as a solid starting point to grasp the inner workings of CUDA and its pivotal role in accelerating the voxelization process. So, let's embark on this informative journey and unravel the secrets behind this powerful technique.

### 2.1. Folder structure

The project directory contains the following files and folders:

```
    ├── CMakeLists.txt
    ├── data
    │   └── test
    │       ├── pc1.bin
    │       └── pc2.bin
    ├── include
    │   ├── common.h
    │   ├── kernel.h
    │   └── preprocess.h
    ├── main.cpp
    ├── README.md
    └── src
        ├── preprocess.cpp
        └── preprocess_kernel.cu
```

- [CMakeLists.txt](#) - CMake configuration file for building the project
- **data** - Directory containing test data used in the project.
  - **test** - Subdirectory containing binary data files.
- **include** - Directory containing header files.
  - [common.h](#) - Header file with common definitions and macros.
  - [kernel.h](#) - Header file with CUDA kernel function declarations.
  - [preprocess.h](#) - Header file for the preprocessing functions.
- [main.cpp](#) - Main C++ source file that orchestrates the voxelization process.
- [README.md](#) - Markdown file containing project documentation and information.
- **src** - Directory containing source files.
  - [preprocess.cpp](#) - Source file with the implementation of preprocessing functions.
  - [preprocess_kernel.cu](#) - CUDA source file with kernel implementations. -->
