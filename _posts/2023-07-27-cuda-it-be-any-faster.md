---
layout: post
title: "CUDA it Be Any Faster?"
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
    - The hash table will store all unique_voxels hashed voxel_offset as keys and the voxel_id for each unique voxel as ID.This means that we only store information about voxels that actually have a point in them. The voxels are flattened out to calculate offset and the hashed value of this flattened out voxel acts as key. The value is the 
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

<br>

## 1. Intuition 
In this section, we will embark on an intuitive exploration of the point cloud voxelization process using CUDA. To that end, let's first set the stage by taking a look at a simple 2D grid. We'll create a 3x3 grid and randomly select 15 points within its boundaries. Some points will also lie slightly outside the grid to make the example more interesting. Visualizing this grid and its sample points, we get the following plot:

<div style="width: 60%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_2/points.jpeg" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The key steps involved in CUDA based Voxelization
</div>
</div>

Now, let us understand one last detail before we begin the actual processing. In the figure shown below, on the left, we have a 2D grid representing the cells with their corresponding indices ranging from (0, 0) to (2, 2). Each cell in the grid is identified by its x and y coordinates, starting from the bottom-left corner and progressing towards the top-right corner. However, from now on, we will refer to this serialized integer index as `voxel_offset`,  which uniquely represents each cell in a sequential order from 0 to 8, as shown on the right side of the figure.
<div style="width: 70%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_2/serialize.jpeg" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The key steps involved in CUDA based Voxelization
</div> 
</div>
This `voxel_offset`` plays a crucial role in the upcoming CUDA-based voxelization process, allowing us to efficiently access and process voxel data in a linear manner. By representing the grid in this serialized format, we can easily map each voxel's position to its corresponding voxel ID, making the hash map implementation more streamlined and intuitive.

The process of converting a point cloud to voxels using CUDA involves three main steps: hash map building, voxelization, and feature extraction as shown in the Figure below. Hash map building efficiently stores information about unique voxels that contain points, eliminating the need to process all grid cells. Voxelization assigns each point to its corresponding voxel, creating a serialized array that stores point features for all voxels. Finally, feature extraction calculates the average features for each voxel, resulting in an efficient representation of point cloud features. In the following section we will understand each of these steps intuitively using our toy example before we delve into the real deal.

<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_2/steps.jpg" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The key steps involved in CUDA based Voxelization
</div>
<br>

### A. Build Hashmaps
Now, let's explore the crucial role of hash maps in the voxelization process. Before diving into the intricacies of building hash maps, it's essential to understand why they are necessary. In our 3x3 grid example, we observed that out of the 9 cells, only 6 cells contain points, while the remaining 3 cells are entirely empty (as marked in red in the figure below). This situation presents a compelling opportunity for optimization, as processing all 9x9 cells would be highly inefficient and computationally wasteful. That's where hash maps comes into play.

<div style="width: 70%;margin: 0 auto;">
<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_2/empty_voxels.jpeg" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The key steps involved in CUDA based Voxelization
</div> 
</div>

Hash maps offer an efficient way to store and access data by associating each voxel's position with its corresponding information. By directly mapping the unique voxel positions as keys( `voxel_offset` as described earlier ) to their respective `voxel_id`s ( will be explained soon) as values, we can efficiently eliminate the need to process all the empty cells. This approach drastically reduces memory consumption and processing time, making voxelization of point clouds significantly faster and more resource-efficient. So, let's now understand how we go from our scattered 2D points, with certain empty cells, to an efficient and compact hashmap : 
1. **Filtering Points**: We begin by filtering out all the points that lie outside the defined boundary. These points are not relevant for our voxelization and can be safely excluded from further processing. 
2. **Voxel Computation** : We take each remaining point from the filtered set and calculate its corresponding voxel indices (voxel_x, voxel_y) using a general point-to-voxel clamping technique. This process ensures that each point is precisely associated with a specific voxel in the 2D grid.

    Let's consider an example point from our 2D grid with coordinates `(1.8, 2.5)`. To convert this point into a voxel, we essentially apply `floor` operation to both the coordinates of the points.
    The actual computation is a little more involved, but we will cover that in later section.

    ```
      Original Point:[1.8, 2.5]
      Clamped Point: [floor(1.8), floor(2.5)] =[1, 2]
    ```

    Now, the clamped point `(1, 2)` represents the voxel indices` (voxel_x, voxel_y)` in the 2D grid. This means that the original point `(1.8, 2.5)` is associated with the voxel at grid position `(1, 2)`. By performing this computation for all points, we establish a one-to-one mapping between each point and its respective voxel in the 2D grid.

3. **Hash Table Insertion**: Next, we start the hash table insertion, by passing the key as `voxel_offset`. The corresponding value for this key is the unique voxel counter, reffered to as `voxel_id`, which counts the current number of voxels being added to the hash table. The hash table is a simple array like data structure, which stores all keys first, followed by their corresponding values as shown in the Figure below. 
<div class="row">
<div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html path="/assets/img/blog/blog_2/hash1.jpg" title="hash table" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
    Initial Hash Table ( of size 12). First we store the keys and then the corresponding values.
</div> 
The insertion essentially happens in three steps :

1. **Hashing the Key (Voxel_Offset):** The first step in the insertion process is to hash the given key, which represents the `voxel_offset`. Hashing is a mathematical function that transforms the key into a unique numeric value. In our example, we use a simple hash function that involves multiplying the `voxel_offset` by 2.

2. **Modulus Operation to Find Slot:** After hashing the key, we apply the `modulus(%)` operation with the size of the hash table divided by 2. This helps us find the slot in the hash table where the key-value pair should be inserted. The modulus operation ensures that the slot index remains within the valid range of the hash table.

3. **Compare and Swap Operation:** The final step is to perform the compare and swap operation (CAS) on the hash table at the slot obtained from the modulus operation. The CAS operation checks if the slot is empty. If it is, the insertion is successful, and the key-value pair is placed in that slot. However, if the slot is not empty, it indicates a collision, meaning that another key with the same hash value already occupies that slot. In this case, we need to handle the collision by employing a linear probing technique, where we check the next slot in the hash table until we find an empty slot. Here we have assumed the size of hash table to be 12, but in practice the size of the hash table is much larger than number of keys, to avoid collisions.

By following these three basic steps, we can efficiently insert points into the hash table, associating each voxel's position with its corresponding unique ID. This enables us to store relevant information about the voxels containing points and optimize the voxelization process. Now, let's dive into an example to illustrate this process in action. 


#### Example 1


```
Psuedo Code : Example 1

INSERTION 1 : Pt(2.3,2.8) 

/* Simple case where the slot for the given key is empty.
We just insert the key to this slot and increment the value of total voxels by 1.*/

Step 1 : Voxel Computation 
voxel_x = floor(2.3) = 2
voxel_y = floor(2.8) = 2
voxel_offset = (voxel_y * size_y) + voxel_x  = 2*3 + 2 = 8 
key = voxel_offset

Step 2 : Hashing of key 
hash_key = hash(key) = key*2 = 8*2 = 16.

Step 3 : Find Slot using Modulus
slot = hash_key%(hash_size/2) = 16%6 = 4

Step 4 : Compare And Swap
value = total_voxels = 0.
// simple case 
hash_table[slot] = key
hash_table[slot+hash_size/2] = value
total_voxel++

```
As seen in the figure below, we simply insert the given key 8, at the computed slot 4, because it was empty. Then we insert the corresponding value at slot 10, (4+6).
The value is zero because this is the first unique voxel being inserted in the hash table.

<div style="width: 70%;margin: 0 auto;">
<div class="row">
<div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html path="/assets/img/blog/blog_2/insert1.jpeg" title="hash table" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
    Example 1 : Hash insert operation for point Pt(2.3,2.8) in red. Inserting key 8 at slot (4) and value 0, at slot (4+6=10). 
</div> 
</div>

#### Example 2

```
Psuedo Code : Example 2

INSERTION 2 : Pt(1.8,0.5) 

/* Simple case where the slot for the given key is empty.
We just insert the key to this slot and increment the value of total voxels by 1.*/

Step 1 : Voxel Computation 
voxel_x = floor(1.8) = 1
voxel_y = floor(0.5) = 0
voxel_offset = (voxel_y * size_y) + voxel_x = 0*3 + 1 = 1
key = voxel_offset

Step 2 : Hashing of key 
hash_key = hash(key) = key*2  = 1*2 = 2.

Step 3 : Find Slot using Modulus
slot = hash_key%(hash_size/2) = 2%6 = 2

Step 4 : Compare And Swap
value = total_voxels = 1.
// simple case 
hash_table[slot] = key
hash_table[slot+hash_size/2] = value
total_voxel++

```

As seen in the figure below, we simply insert the given key 1, at the computed slot 2, because it was empty. Then we insert the corresponding value at slot 8, (2+6).
The value is the count of the current voxels, which is 1.

<div style="width: 70%;margin: 0 auto;">
<div class="row">
<div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html path="/assets/img/blog/blog_2/insert2.jpeg" title="hash table" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
     Example 2 : Hash insert operation for point Pt(1.8,0.5) shown in red. Inserting key 1 at slot (2) and value 0, at slot (2+6=8). 
</div> 
</div>

#### Example 3

```
Psuedo Code : Example 3 

INSERTION 3 : Pt(1.7,2.8) 

/*Collision Case wherein the slot computed uusing hash key is already filled
 up by a different key. In this case, insert the key in the next free slot. */

Step 1 : Voxel Computation 
voxel_x = floor(1.7) = 1
voxel_y = floor(2.8) = 2
voxel_offset = (voxel_y * size_y) + voxel_x = 2*3 + 1 = 7
key = voxel_offset

Step 2 : Hashing of key 
hash_key = hash(key) = key*2 =  7*2 = 14.

Step 3 : Find Slot using Modulus
slot = hash_key%(hash_size/2) = 14%6 = 2

Step 4 : Compare And Swap
value = total_voxels = 2
// collision case
hash_table[slot]!= Empty && hash_table[slot] != voxel_offset
// insert in next slot ( if free)
hash_table[++slot] = key
hash_table[slot+hash_size/2] = value
total_voxel++

```
Now we look at a more interesting example where the slot computed using the modulus of the hashed key is already present in the hash table. So a collision occurs, because the slot is 
already filled up by a different key. We look at this case while trying to insert a Point (1.7,2.8). In such a case of collision, we simply insert the key in the next free slot, which in this case is slot (3), as shown below.

<div style="width: 70%;margin: 0 auto;">
<div class="row">
<div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html path="/assets/img/blog/blog_2/collision.jpeg" title="hash table" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
     Example 3 : Hash insert operation for point Pt(1.7,2.8) shown in red. The slot (2) already has key 1 present, which is different from key 7. This indicates a collision, we insert in 
     in the next empty slot which is slot 3. This is called linear probing.
</div> 
</div>

#### Example 4

```
Psuedo Code : Example 4

INSERTION 4 : Pt(1.8,0.5) 

/* Already inserted case where the slot for the given key is not emppty.
 No insertion happens, since slot already has same key. */

Step 1 : Voxel Computation 
voxel_x = floor(1.8) = 1
voxel_y = floor(0.5) = 0
voxel_offset = (voxel_y * size_y) + voxel_x = 0*3 + 1 = 1
key = voxel_offset

Step 2 : Hashing of key 
hash_key = hash(key) = key*2  = 1*2 = 2.

Step 3 : Find Slot using Modulus
slot = hash_key%(hash_size/2) = 2%6 = 2

Step 4 : Compare And Swap
value = total_voxels = 3.
// already inserted case 
hash_table[slot] != Empty && hash_table[slot] == key
// no insertion.

```

As seen in the figure below, we skip insertion, at the computed slot 2, because it already had the current points key 1. The goal is to only store unique `voxel_offset` as key
and the number of such unique voxels as values. Since this `voxel_offset` has already been inserted in the hash table and is not unique, we simply skip insertion.

<div style="width: 70%;margin: 0 auto;">
<div class="row">
<div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html path="/assets/img/blog/blog_2/already-inserted.jpeg" title="hash table" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
     Example 4 : Hash insert operation for point Pt(1.3,0.3) shown in red.
     No insertion happens, since slot already has same key.  
</div> 
</div>
After successfully completing the hash insertion step, we have effectively built a hash table that stores the `voxel_offset` (representing the flattened voxel position) as keys and the corresponding `voxel_ID` as values. The `voxel_ID` is an indicator of the total number of unique voxels, which is updated as we encounter new unique voxels during the hash table construction process. This data structure offers two significant advantages. Firstly, it allows us to process only those voxels that contain points, thereby optimizing our further steps. Additionally, the process of mapping a point to its respective voxel becomes a constant-time operation `O(1)` due to the efficient key searching capability of the hash table. This constant-time retrieval is significantly faster compared to traditional "dictionary(map)" structures, which involve `O(log(n))` time complexity for key searches.
<br>
<br>

### B. Voxelization
<br>

<div class="row">
    <div class="col-sm mt-3 mt-md-0 text-center"> <!-- Add 'text-center' class here -->
        {% include figure.html path="/assets/img/blog/blog_2/steps.jpg" title="latency compare" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The key steps involved in CUDA based Voxelization
</div>

Having accomplished the challenging task of creating the hash table, we now move on to the second step: "Voxelization." In this step, we utilize the hash table to efficiently process every point in a separate thread. The objective is to construct a 3D array called `voxel_temp` with a size of (`max_voxels` * `max_points_per_voxel` * `num_features`), where all point features for each voxel are stored serially. Points belonging to the same voxel are grouped together in memory, thereby optimizing data access and manipulation.We simplify the process by breaking it down into three logical steps.
    
1. **Compute Voxel Offset from Point**
For each point, we determine the corresponding voxel offset. The voxel offset represents a unique identifier for a specific voxel in our 3D grid. Imagine each voxel as a small box in our 3x3 grid, and the voxel offset as a label that tells us which box this point belongs to.

2. **Efficiently Find Voxel ID from the Hash Table**
To quickly locate the voxel's position in the array, we leverage the previously constructed hash table. Using the voxel offset as the key, we perform a constant-time search in the hash table to find the corresponding value, which represents the unique voxel ID. Think of this process as instantly finding the box's label (voxel ID) when given the box's unique identifier (voxel offset).

3. **Store Point Features in the Voxel Array**
With the voxel ID in hand, we efficiently store the point's features in the voxel_temp array. This array is designed to hold all the point features for every voxel in a serialized manner, ensuring that points from the same voxel are stored together. We use the voxel ID to determine the correct position in the array to store this point's features, allowing us to efficiently group all points for each voxel.



<div style="width: 80%;margin: 0 auto;">
<div class="row">
<div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html path="/assets/img/blog/blog_2/voxel.jpg" title="hash table" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
     Voxelization :  Points A and B in voxel with voxel_offset=5 are stored in position 4 in voxel_temp array. This position 4 is derived from the hash table by looking up value corresponding to the key 5.
</div> 
</div>

Let's dive into an example to better understand the Voxelization process. Consider our 3x3 grid, and we have two points: `Point A(2.9, 1.7)` and `Point B(2.2, 1.3)`.

```
Step 1: Compute Voxel Offset from Points
- For Point A, we calculate the voxel offset by considering its position in the grid.
- As Point A falls within the voxel at (2, 1), the voxel offset for Point A becomes 5. 
- Similarly, for Point B, the voxel offset is also 5 as it belongs to the same voxel.

Step 2: Find Voxel ID from the Hash Table
- We refer to our previously constructed hash table to quickly determine the voxel_ID 
  associated with voxel_offset 5.
- Upon checking the hash table, we find that the value corresponding to the key 5 is 4
- This indicates that this voxel has been assigned voxel ID 4.

Step 3: Store Point Features in the Voxel Temp Array
- Now that we have the voxel ID (4), we store the point features of Point A and Point B 
  in the voxel_temp array at the 4th position. 
- The voxel_temp array contains information about all points for each voxel, 
  ensuring that points within the same voxel are grouped together.

``` 

In the accompanying Figure above, we visually depict this process. We show the 3x3 grid, with `Point A` and` Point B` marked inside the voxel they belong to. Next, we present the hash map, where we highlight` voxel_offset (5)` and` voxel_id (4)` to showcase how they are linked. Subsequently, we display the `voxel_temp` array, with 6 voxels filled up since only 6 of the 9 voxels in our 3x3 grid have points. Finally, we zoom into the voxel with `voxel_id` 4 to witness the two points, A and B, stored serially in this array with their respective `(x,y)` values.

This example helps illustrate how the Voxelization process organizes point data efficiently, grouping all points belonging to each voxel together, thanks to the hash map's guidance. This method significantly speeds up access and processing of point cloud data, making voxel-based approaches highly effective for a wide range of applications.

In summary, after completing the Voxelization step, we achieve an organized arrangement where all points belonging to each voxel are conveniently stored together. The hash table plays a key role, acting as a guide that allows us to locate each voxel's position in the array with ease. 

<br>

### C. Feature extraction
In the final step of our voxelization process, known as Feature Extraction, we aim to extract meaningful information from the `voxel_temp` array, which contains all point features grouped by their respective voxel IDs. The goal is to compute average feature values for each voxel and store them in a `voxel_features` array.


1. **Prepare for Feature Extraction**
The feature extraction process operates on a per-voxel basis, where each voxel's features are processed independently. To begin, we initialize the num_points_per_voxel array to keep track of the number of points in each voxel. Then, for each voxel, we iterate over its points to calculate the total number of points and update the corresponding entry in the `num_points_per_voxel` array.

2. **Calculate Average Feature Values**
Next, we calculate the average feature values for each voxel. Starting with the first point in the `voxel_temp` array for a given voxel, we compute the offset to access this voxel's data in the array. For subsequent points within the same voxel, we iterate over all features (e.g., x, y, z, intensity, time) and sum up their values.

3. **Update Voxel Features**
After summing up the features for all points within the voxel, we calculate the average by dividing the sum by the total number of points in that voxel. These averaged feature values are then updated in the `voxel_features` array at the position corresponding to the voxel's ID.

In summary, the Feature Extraction step processes `voxel_temp` data to calculate the average feature values for each voxel. This process is performed in parallel for all voxels, utilizing one thread per voxel. By the end of this step, the `voxel_features` array holds crucial information about each voxel's characteristics in continuous memory, ready for further analysis and applications in point cloud processing. For example we see feature extraction for the voxel at position 4 as shown in the Figure below.


<div style="width: 80%;margin: 0 auto;">
<div class="row">
<div class="col-sm mt-3 mt-md-0 text-center">
    {% include figure.html path="/assets/img/blog/blog_2/feature_extract.jpg" title="hash table" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">
     Voxelization :  Points A and B in voxel with voxel_offset=5 are stored in position 4 in voxel_temp array. This position 4 is derived from the hash table by looking up value corresponding to the key 5.
</div> 
</div>

```
Voxel Feature Extraction for Voxel at Position 4

1. Prepare for Feature Extraction:
   We first initialize the num_points_per_voxel array to keep 
   track of the number of points in each voxel.
   For the voxel at position 4, num_points_per_voxel[4] = 2, as it contains two points.

2. Calculate Average Feature Values:
   We now iterate over the points in the voxel_temp array 
   for the voxel at position 4.Let's consider the features x and y 
   for this example.

   Iteration 1:
   - Point A: x = 2.9, y = 1.7
   - Sum of x = 2.9, Sum of y = 1.7

   Iteration 2:
   - Point B: x = 2.2, y = 1.3
   - Sum of x = 2.9 + 2.2 = 5.1, Sum of y = 1.7 + 1.3 = 3.0

3. Update Voxel Features:
   After iterating through all points in the voxel, we have the 
   sums of x and y for each feature.Now, we calculate the average by dividing
   the sum by the total number of points in the voxel (num_points_per_voxel[4] = 2).

   - Average x = 5.1 / 2 = 2.55
   - Average y = 3.0 / 2 = 1.5

   Finally, we update the voxel_features array for the voxel at position 4 with 
   the calculated average feature values: voxel_features[4] = (2.55, 1.5)

```





## 2. Inside the Code
In this section, we will take a comprehensive look at the CUDA-powered voxelization implementation. We'll dissect the code, step by step, to understand the magic behind its blazing-fast performance. To begin our exploration, let's first examine the folder structure that forms the foundation of this efficient voxelization engine. Understanding the organization of the code will serve as a solid starting point to grasp the inner workings of CUDA and its pivotal role in accelerating the voxelization process. So, let's embark on this informative journey and unravel the secrets behind this powerful technique.

#### 2.1 Setting up Locally
Before diving into the code, let's set up the project locally by following these steps:
Setup Project Locally:

1. Install CUDA > 11.4.
2. Ensure you are using Ubuntu > 18.04.
3. Add CUDA path to `PATH` and `LD_LIBRARY_PATH`.


Now, let's build and run the project with the provided commands:
<br>


```bash
git clone https://github.com/sanket-pixel/voxelize-cuda
cd voxelize-cuda
mkdir build && cd build
cmake ..
make
./voxelize_cuda ../data/test/
```
You can expect an output similar to this:

```
GPU has cuda devices: 1
----device id: 0 info----
  GPU : GeForce RTX 2060 
  Capability: 7.5
  Global memory: 5912MB
  Const memory: 64KB
  SM in a block: 48KB
  Warp size: 32
  Threads in a block: 1024
  Block dimension: (1024,1024,64)
  Grid dimension: (2147483647,65535,65535)

Total 2

<<<<<<<<<<<
Load file: ../data/test/291e7331922541cea98122b607d24831.bin
Find points num: 239911
[TIME] Voxelization: 4.66307 ms
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/test/3615d82e7e8546fea5f181157f42e30b.bin
Find points num: 267057
[TIME] Voxelization: 2.34752 ms
>>>>>>>>>>>

```
Once you have obtained this output, you can take a cup of coffee, as we are now ready to deep dive into the code. Let's explore the implementation in detail.

#### 2.2. Folder structure

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
  - [preprocess_kernel.cu](#) - CUDA source file with kernel implementations.


#### 2.3 Code Walkthrough

The main entry point of the program is the main function in the `main.cpp` file. It starts by checking the command-line arguments and loading the point cloud data from the specified folder.

##### Step 1: Setup and Initialization

- The `GetDeviceInfo` function is used to print information about the CUDA devices available on the system.

- The `getFolderFile` function is used to get a list of files in the specified data folder with a ".bin" extension.

- The `loadData` function is used to load the binary data file into memory.

##### Step 2: Preprocessing

The preprocessing is handled by the `PreProcessCuda` class, defined in the `preprocess.h` and `preprocess.cpp` files.
It performs three main operations: hash map building, voxelization, and feature extraction.

**A. Hash Map Building**:

The hash map building is performed in the `buildHashKernel` CUDA kernel defined in the `preprocess_kernel.cu` file. This kernel takes the input point cloud data and converts it into voxel coordinates using the specified voxel size and range. It then builds a hash table that maps each voxel offset to its corresponding voxel ID.

**B. Voxelization**:

The voxelization is performed in the `voxelizationKernel` CUDA kernel, also defined in the `preprocess_kernel.cu` file. This kernel uses the hash table built in the previous step to assign each point to its corresponding voxel. It counts the number of points in each voxel and stores them in the `num_points_per_voxel` array. It also serializes the point features for each voxel in the `voxels_temp` array.

**C. Feature Extraction**:

The feature extraction is handled by the `featureExtractionKernel` CUDA kernel, also defined in the `preprocess_kernel.cu` file. This kernel takes the serialized point features in the `voxels_temp` array and computes the average feature values for each voxel. It stores the averaged features in the `voxel_features` array.

##### Step 3: Output and Cleanup

After the preprocessing is complete for all the input files, the program outputs the results and frees the allocated memory.

#### 2.4 Deep Dive
Now that we have taken a closer look at the basic walkthrough of the code, let's embark on a more comprehensive deep dive, exploring the intricacies of the CUDA kernels and delving into the inner workings of the preprocessor class. We will gradually progress from the core CUDA kernel, which handles the voxelization process efficiently through parallelism, to the preprocessor class, where these kernels are utilized. Finally, we will uncover how the main cpp file leverages the functionalities of the preprocessor class to apply voxelization on the input point cloud data, culminating in the generation of 3D voxels. This step-by-step approach will allow us to understand how each component contributes to the overall process and how they interact harmoniously to produce the desired output. So, let's begin our journey from inside to out, unraveling the complexities of the code and gaining a deeper understanding of its functioning.

**1. CUDA Kernel for Building HashTables**

The `buildHashKernel` is a CUDA kernel that performs the hash table building process. It takes the input point cloud data (`points`) and converts each point into voxel coordinates based on the specified voxel size and range. Then, it calls the `insertHashTable` function to insert each voxel offset as a key into the hash table, and the `real_voxel_num` variable is updated to keep track of the number of unique voxels in the hash table. The `buildHashKernel` function is executed by multiple CUDA threads in parallel, each processing a different point from the input point cloud. As a result, the hash table is efficiently built using GPU parallelism, and each voxel offset is uniquely assigned to an entry in the hash table.

```cpp

__global__ void buildHashKernel(const float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size,
        int grid_y_size, int grid_x_size, int feature_num,
	unsigned int *hash_table, unsigned int *real_voxel_num) {
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) {
    return;
  }
  
  float px = points[feature_num * point_idx];
  float py = points[feature_num * point_idx + 1];
  float pz = points[feature_num * point_idx + 2];

  if( px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range
    || pz < min_z_range || pz >= max_z_range) {
    return;
  }

  unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
  unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
  unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
  unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size
	                    + voxel_idy * grid_x_size
                            + voxel_idx;
  insertHashTable(voxel_offset, real_voxel_num, points_size * 2 * 2, hash_table);
}

```

Here's how the buildHashKernel works:
0. `__global__ void buildHashKernel(...)`: This line defines the `buildHashKernel` function as a CUDA kernel using the `__global__` function modifier. As a kernel, this function will be executed in parallel by multiple threads on the GPU.

1. `int point_idx = blockIdx.x * blockDim.x + threadIdx.x`: This line calculates the index of the current point to be processed by the CUDA thread.

2. `if (point_idx >= points_size) { return; }`: This condition checks if the thread index is out of bounds (i.e., beyond the number of points in the input points array). If so, the thread returns early to avoid processing invalid data.

3. `float px = points[feature_num * point_idx]; ...`: These lines extract the X, Y, and Z coordinates of the current point from the input points array based on the `feature_num` (the number of features per point).

4. `if (px < min_x_range \|\| px >= max_x_range \|\| ...`: This condition checks if the current point lies within the specified 3D range (min/max X, Y, Z). If the point is outside this range, it is not considered for voxelization, and the thread returns early.

5. `unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size); ...`: These lines calculate the voxel coordinates (`voxel_idx`, `voxel_idy`, `voxel_idz`) corresponding to the current point's X, Y, and Z coordinates based on the specified voxel sizes and ranges.

6. `unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size ...`: This line calculates the `voxel_offset` based on the voxel coordinates. The `voxel_offset` is a flattened  index for each voxel within the 3D grid.

7. `insertHashTable(voxel_offset, real_voxel_num, points_size * 2 * 2, hash_table);`:  This line calls the `insertHashTable` function to insert the current `voxel_offset` into the hash table. It also updates the `real_voxel_num` variable using the `atomicAdd` function to keep track of the number of unique voxels added to the hash table.


**2. Inserting into HashTable**

Next, let's move on to the `insertHashTable` function:

```cpp
// Function to insert a key-value pair into the hash table
__device__ inline void insertHashTable(const uint32_t key, uint32_t *value,
		const uint32_t hash_size, uint32_t *hash_table) {
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2)/*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  while (true) {
    uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key);
    if (pre_key == empty_key) {
      hash_table[slot + hash_size / 2 /*offset*/] = atomicAdd(value, 1);
      break;
    } else if (pre_key == key) {
      break;
    }
    slot = (slot + 1) % (hash_size / 2);
  }
}
```

Explanation:

The `insertHashTable` function is responsible for inserting a key-value pair into the hash table. It uses the previously explained `hash` function to compute the hash value of the key, and then it resolves hash collisions using a technique called linear probing.

Here's how the `insertHashTable` function works:

1. `uint64_t hash_value = hash(key)`: This line computes the hash value of the input `key` using the `hash` function.

2. `uint32_t slot = hash_value % (hash_size / 2)`: This line calculates the initial slot index in the hash table by taking the modulo of the hash value with half of the hash table size. This ensures that the slot index is within the valid range of the hash table.

3. `uint32_t empty_key = UINT32_MAX`: This line sets the `empty_key` variable to the maximum value of a 32-bit unsigned integer. This value is used to indicate an empty slot in the hash table.

4. `while (true) { ... }`: This is a loop that continues until the key is successfully inserted into the hash table. It handles hash collisions using linear probing.

5. `uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key)`: This line performs an atomic compare-and-swap operation (CAS) on the hash table. It checks if the slot at the current index (`slot`) is empty (i.e., contains `empty_key`). If it is empty, it atomically swaps the value with the input `key`, effectively inserting the key into the hash table.

6. `if (pre_key == empty_key) { ... }`: This condition checks if the CAS operation was successful, indicating that the key was inserted into the hash table. If successful, the function proceeds to update the offset in the hash table for the corresponding value (used in feature extraction).

7. `hash_table[slot + hash_size / 2 /*offset*/] = atomicAdd(value, 1)`: This line atomically increments the value in the hash table at the offset `slot + hash_size / 2`, effectively storing the value for the given key. The `atomicAdd` function ensures that multiple threads trying to insert the same key concurrently will get unique values.

8. `else if (pre_key == key) { ... }`: This condition handles the case when the slot already contains the same key (a duplicate). In this case, the function breaks out of the loop, as there's no need to insert the key again.

9. `slot = (slot + 1) % (hash_size / 2)`: This line updates the slot index using linear probing by incrementing it by 1 and wrapping around to the beginning if it exceeds half of the hash table size.

The `insertHashTable` function plays a crucial role in building the hash table, which is later used in voxelization and feature extraction.


**3. Hash Function**

Sure, let's start by explaining the base function, which is the `hash` function:

```cpp
// Hash function for generating a 64-bit hash value
__device__ inline uint64_t hash(uint64_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;
}
```

Explanation:

The `hash` function is a simple hash function that takes a 64-bit integer `k` as input and generates a 64-bit hash value using bitwise operations and multiplication with constants.

Here's how the hash function works:

1. `k ^= k >> 16`: This line performs a bitwise XOR operation between `k` and `k` right-shifted by 16 bits. This step introduces a level of randomness to the bits.

2. `k *= 0x85ebca6b`: This line multiplies `k` by a constant value (`0x85ebca6b`). Multiplication helps in spreading out the bits and reducing collisions.

3. `k ^= k >> 13`: This line again performs a bitwise XOR operation between `k` and `k` right-shifted by 13 bits. This step further increases the randomness of the bits.

4. `k *= 0xc2b2ae35`: This line multiplies `k` by another constant value (`0xc2b2ae35`) to spread out the bits even more.

5. `k ^= k >> 16`: Finally, this line performs a bitwise XOR operation between `k` and `k` right-shifted by 16 bits. This step is the last step of mixing the bits to produce the final hash value.

The `hash` function is used in the hash table building process to generate unique hash values for the voxel offsets in the point cloud data.


**4. Voxelization Kernel**

In the voxelizationKernel CUDA kernel, each thread processes an individual point from the input point cloud. The goal of this function is to efficiently convert the points into their corresponding 3D voxel representations. It first calculates the voxel coordinates for each point based on the specified voxel sizes and ranges. Then, it uses a hash table lookup to find the corresponding voxel ID for each voxel offset. If the voxel ID is within the specified maximum number of voxels, the function atomically adds the point to the voxel and updates the voxel indices accordingly. This ensures a fast and optimized voxelization process for large and sparse point clouds.

```cpp
__global__ void voxelizationKernel(const float *points, size_t points_size,
                                   float min_x_range, float max_x_range,
                                   float min_y_range, float max_y_range,
                                   float min_z_range, float max_z_range,
                                   float voxel_x_size, float voxel_y_size, float voxel_z_size,
                                   int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
                                   int max_points_per_voxel,
                                   unsigned int *hash_table, unsigned int *num_points_per_voxel,
                                   float *voxels_temp, unsigned int *voxel_indices, unsigned int *real_voxel_num) {
  
  // give every point to a thread. Find the index of the current point within this kernel.
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) {
    return;
  }

  // points is the array of points in the point cloud storing point features (x, y, z, intensity, t)
  // in a serialized format. We access px, py, pz of this point now.
  // feature_num is 5, representing the number of features per point.
  float px = points[feature_num * point_idx];
  float py = points[feature_num * point_idx + 1];
  float pz = points[feature_num * point_idx + 2];

  // If the point is outside the range along the (x, y, z) dimensions, stop further 
  // processing and return.
  if (px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range
    || pz < min_z_range || pz >= max_z_range) {
    return;
  }

  // Now find the voxel id for this point using the usual voxel conversion logic.
  unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
  unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
  unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
  // Now find the voxel offset, which is the index of the voxel if all voxels are flattened.
  unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size
                            + voxel_idy * grid_x_size
                            + voxel_idx;
  // We perform a scatter operation to voxels, and the result is stored in 'voxel_id'.
  unsigned int voxel_id = lookupHashTable(voxel_offset, points_size * 2 * 2, hash_table);
  // If the current voxel id is greater than max_voxels, simply return.
  if (voxel_id >= max_voxels) {
    return;
  }
  
  // With the voxel id, we can now atomically increment the counter for the number of points in the voxel.
  unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);

  // If the current number of points in the voxel exceeds the maximum allowed points per voxel, we return.
  if (current_num >= max_points_per_voxel) {
    return;
  }

  // Now we can proceed to add the current point to the voxel's feature list.
  // Calculate the destination offset where the point's features will be stored in the voxels_temp array.
  unsigned int dst_offset = voxel_id * (feature_num * max_points_per_voxel) + current_num * feature_num;
  // Calculate the source offset of the current point's features in the points array.
  unsigned int src_offset = point_idx * feature_num;
  
  // Copy the point's features from the points array to the corresponding location in the voxels_temp array.
  // This effectively adds the point's features to the voxel's feature list.
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
    voxels_temp[dst_offset + feature_idx] = points[src_offset + feature_idx];
  }
  
  // Store additional information about the voxel (its indices along X, Y, Z axes) for later processing.
  uint4 idx = {0, voxel_idz, voxel_idy, voxel_idx};
  ((uint4 *)voxel_indices)[voxel_id] = idx;

}
```

Explanation:

Here's how the `voxelizationKernel` works:

1. `int point_idx = blockIdx.x * blockDim.x + threadIdx.x;`: This line calculates the index of the current point to be processed by the CUDA thread.

2. `if (point_idx >= points_size) { return; }`: This condition checks if the thread index is out of bounds, i.e., beyond the number of points in the input points array. If so, the thread returns early to avoid processing invalid data.

3. `float px = points[feature_num * point_idx]; float py = points[feature_num * point_idx + 1]; float pz = points[feature_num * point_idx + 2];`: These lines extract the X, Y, and Z coordinates of the current point from the input points array based on the `feature_num` (the number of features per point).

4. `if (px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range || pz < min_z_range || pz >= max_z_range) { return; }`: This condition checks if the current point lies within the specified 3D range (min/max X, Y, Z). If the point is outside this range, it is not considered for voxelization, and the thread returns early.

5. `unsigned int voxel_idx = floorf((px - min_x_range) / voxel_x_size); unsigned int voxel_idy = floorf((py - min_y_range) / voxel_y_size); unsigned int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);`: These lines calculate the voxel coordinates (`voxel_idx`, `voxel_idy`, `voxel_idz`) corresponding to the current point's X, Y, and Z coordinates based on the specified voxel sizes and ranges.

6. `unsigned int voxel_offset = voxel_idz * grid_y_size * grid_x_size + voxel_idy * grid_x_size + voxel_idx;`: This line calculates the voxel offset based on the voxel coordinates. The voxel offset is a unique identifier for each voxel within the 3D grid.

7. `unsigned int voxel_id = lookupHashTable(voxel_offset, points_size * 2 * 2, hash_table);`: This line calls the `lookupHashTable` function to find the corresponding voxel ID for the current voxel offset using the hash table.

8. `if (voxel_id >= max_voxels) { return; }`: This condition checks if the current voxel ID is greater than or equal to `max_voxels`, indicating that the maximum number of allowed voxels has been reached. If so, the thread returns early.

9. `unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);`: This line uses the `atomicAdd` function to atomically increment the number of points in the voxel represented by `voxel_id` in the `num_points_per_voxel` array.

10. `if (current_num < max_points_per_voxel) { ... }`: This condition checks if the current number of points in the voxel is less than the maximum allowed per voxel. If so, the thread proceeds to add the current point's features to the voxel.

11. `unsigned int dst_offset = voxel_id * (feature_num * max_points_per_voxel) + current_num * feature_num; unsigned int src_offset = point_idx * feature_num;`: These lines calculate the offsets for copying the current point's features to the voxel in the `voxels_temp` array.

12. `for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) { voxels_temp[dst_offset + feature_idx] = points[src_offset + feature_idx]; }`: This loop copies the features of the current point to the appropriate location in the `voxels_temp` array, effectively adding the point to the voxel.

13. `uint4 idx = {0, voxel_idz, voxel_idy, voxel_idx}; ((uint4 *)voxel_indices)[voxel_id] = idx;`: These lines create an index vector (`idx`) containing information about the voxel's position in the grid and store it in the `voxel_indices` array at the location corresponding to `voxel_id`. This allows quick lookup of voxel information during subsequent processing.


The `voxelizationKernel` efficiently assigns each point to its corresponding voxel, ensuring that points are appropriately added to the voxel's feature list without exceeding the maximum allowed points per voxel.


**5. Voxelization Launch**

The `voxelizationLaunch` function is a key step in the voxelization process. It is responsible for launching two CUDA kernels: `buildHashKernel` and `voxelizationKernel`. Let's break down the function and its components:


```cpp
cudaError_t voxelizationLaunch(const float *points, size_t points_size, float min_x_range, float max_x_range,
                          float min_y_range, float max_y_range,float min_z_range, float max_z_range,
                          float voxel_x_size, float voxel_y_size, float voxel_z_size, int grid_y_size,
                          int grid_x_size, int feature_num, int max_voxels,int max_points_per_voxel,
                          unsigned int *hash_table, unsigned int *num_points_per_voxel,float *voxel_features, 
                          unsigned int *voxel_indices,unsigned int *real_voxel_num, cudaStream_t stream) 
      {
    // how many threads in each block
    int threadNum = THREADS_FOR_VOXEL;
    // how many blocks needed if each point gets on thread.
    dim3 blocks((points_size+threadNum-1)/threadNum);
    // how many threads in each block
    dim3 threads(threadNum);
    // how many blocks needed to launch the kernel, how many threads in each block,
    // how many bytes for dynamic shared memory  ( zero here), cuda stream
    buildHashKernel<<<blocks, threads, 0, stream>>>
      (points, points_size,
          min_x_range, max_x_range,
          min_y_range, max_y_range,
          min_z_range, max_z_range,
          voxel_x_size, voxel_y_size, voxel_z_size,
          grid_y_size, grid_x_size, feature_num, hash_table,
    real_voxel_num);
    voxelizationKernel<<<blocks, threads, 0, stream>>>
      (points, points_size,
          min_x_range, max_x_range,
          min_y_range, max_y_range,
          min_z_range, max_z_range,
          voxel_x_size, voxel_y_size, voxel_z_size,
          grid_y_size, grid_x_size, feature_num, max_voxels,
          max_points_per_voxel, hash_table,
    num_points_per_voxel, voxel_features, voxel_indices, real_voxel_num);
    cudaError_t err = cudaGetLastError();
    return err;
}
```
  Explanation:
Here's how the `voxelizationLaunch` function works:

1. `int threadNum = THREADS_FOR_VOXEL;`: This line sets the number of threads per block for the CUDA kernel. The value is obtained from the constant `THREADS_FOR_VOXEL`, which likely represents an optimal number of threads for efficient computation.

2. `dim3 blocks((points_size+threadNum-1)/threadNum);`: This line calculates the number of blocks needed to launch the kernel based on the total number of points (`points_size`) and the `threadNum`. It ensures that all points are processed by the threads efficiently.

3. `dim3 threads(threadNum);`: This line sets the number of threads in each block based on the previously calculated `threadNum`.

4. `buildHashKernel<<<blocks, threads, 0, stream>>>(...)`: This line launches the `buildHashKernel` CUDA kernel. It processes the input points to build the hash table, which maps voxel offsets to voxel IDs.

5. `voxelizationKernel<<<blocks, threads, 0, stream>>>(...)`: This line launches the `voxelizationKernel` CUDA kernel. It voxelizes the input points based on the computed hash table, assigning points to corresponding voxels and storing the voxel features.

6. `cudaError_t err = cudaGetLastError(); return err;`: These lines check for any errors that occurred during kernel launches. If there are any errors, they will be returned by the function, indicating a problem in the GPU computation.

The `voxelizationLaunch` function serves as the entry point to initiate the voxelization process on the GPU. It efficiently divides the data into blocks and threads, launches the necessary CUDA kernels (`buildHashKernel` and `voxelizationKernel`), and checks for any errors in the GPU computation. By effectively utilizing the GPU's parallel processing capabilities, voxelization of large point clouds can be done efficiently and quickly.

Overall, the `voxelizationLaunch` function is a crucial step in the voxelization process, coordinating the parallel execution of the CUDA kernels to efficiently process and voxelate the input point cloud data.


**6. Feature Extraction Kernel**
```cpp
__global__ void featureExtractionKernel(float *voxels_temp,
                                        unsigned int *num_points_per_voxel,
                                        int max_points_per_voxel, int feature_num, half *voxel_features) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    num_points_per_voxel[voxel_idx] = num_points_per_voxel[voxel_idx] > max_points_per_voxel ?
                                      max_points_per_voxel : num_points_per_voxel[voxel_idx];

    int valid_points_num = num_points_per_voxel[voxel_idx];

    int offset = voxel_idx * max_points_per_voxel * feature_num;

    for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
        for (int point_idx = 0; point_idx < valid_points_num - 1; ++point_idx) {
            voxels_temp[offset + feature_idx] += voxels_temp[offset + (point_idx + 1) * feature_num + feature_idx];
        }
        voxels_temp[offset + feature_idx] /= valid_points_num;
    }

    for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
        int dst_offset = voxel_idx * feature_num;
        int src_offset = voxel_idx * feature_num * max_points_per_voxel;
        voxel_features[dst_offset + feature_idx] = __float2half(voxels_temp[src_offset + feature_idx]);
    }
}

```

Here's how the `featureExtractionKernel` works:

1. `int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;`: This line calculates the index of the current voxel to be processed by the CUDA thread. Each CUDA thread corresponds to one voxel, and the `voxel_idx` uniquely identifies the voxel.

2. `num_points_per_voxel[voxel_idx] = num_points_per_voxel[voxel_idx] > max_points_per_voxel ? max_points_per_voxel : num_points_per_voxel[voxel_idx];`: This line checks if the number of points in the current voxel exceeds the `max_points_per_voxel`. If it does, it clips the value to ensure that the feature extraction is performed on a maximum of `max_points_per_voxel` points for each voxel.

3. `int valid_points_num = num_points_per_voxel[voxel_idx];`: This line retrieves the actual number of valid points in the current voxel, which may have been clipped in the previous step.

4. `int offset = voxel_idx * max_points_per_voxel * feature_num;`: This line calculates the offset for accessing the voxel's features in the `voxels_temp` array. It represents the index from which the current voxel's features start in the `voxels_temp` array.

5. The goal of feature extraction is to take the average for each feature (x, y, z, intensity, time) of every point in the voxel. The next few lines of code achieve this by iterating over each feature and each point in the voxel, summing up the feature values, and then dividing the sum by the number of valid points to obtain the average value.

6. `for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) { ... }`: This loop iterates over each feature (x, y, z, intensity, time) and calculates the average value for that feature in the current voxel.

7. The next loop within the feature extraction loop iterates over each point in the voxel (`point_idx`), starting from the second point (index 1) since the first point's feature values were already added to `voxels_temp`.

8. The feature values of each point are added to the corresponding feature in `voxels_temp`. After the loop, `voxels_temp[offset + feature_idx]` contains the sum of feature values for all points in the current voxel for the given feature.

9. Finally, the sum for each feature is divided by the `valid_points_num` to calculate the average feature value for the current voxel. This average feature value is then stored in `voxels_temp` in the same location where the sum was stored earlier.

10. The next loop moves the averaged voxel features from `voxels_temp` to the `voxel_features` array, ensuring that the features for each voxel are stored contiguously in `voxel_features`. The features are converted to the "half" data type (`__float2half`) for memory efficiency.

In summary, the `featureExtractionKernel` takes each voxel represented by a CUDA thread and calculates the average feature values (x, y, z, intensity, time) for all valid points in that voxel. The averaged voxel features are then stored in the `voxel_features` array, which represents the final output of the feature extraction process. The GPU's parallel processing capabilities are utilized to efficiently perform this feature extraction on multiple voxels simultaneously, speeding up the overall computation for large point clouds.


**7. Feature Extraction Launch**

```cpp

cudaError_t featureExtractionLaunch(float *voxels_temp, unsigned int *num_points_per_voxel,
        const unsigned int real_voxel_num, int max_points_per_voxel, int feature_num,
	half *voxel_features, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((real_voxel_num + threadNum - 1) / threadNum);
  dim3 threads(threadNum);
  featureExtractionKernel<<<blocks, threads, 0, stream>>>
    (voxels_temp, num_points_per_voxel,
        max_points_per_voxel, feature_num, voxel_features);
  cudaError_t err = cudaGetLastError();
  return err;
}

```
The featureExtractionLaunch function is the launch function for the featureExtractionKernel, responsible for processing voxel data and extracting features. It takes the necessary input arrays, determines the block and thread configuration based on the number of voxels, launches the kernel, and captures any CUDA errors that might occur during execution.


**8.Generate Voxels**
This function in the `preprocess.cpp` file is responsible for performing voxelization and feature extraction on a set of input points using CUDA on the GPU. Here's how it works:

```cpp
int PreProcessCuda::generateVoxels(const float *points, size_t points_size, cudaStream_t stream)
{
    // flash memory for every run 
    checkCudaErrors(cudaMemsetAsync(hash_table_, 0xff, hash_table_size_, stream));
    checkCudaErrors(cudaMemsetAsync(voxels_temp_, 0xff, voxels_temp_size_, stream));

    checkCudaErrors(cudaMemsetAsync(d_voxel_num_, 0, voxel_num_size_, stream));
    checkCudaErrors(cudaMemsetAsync(d_real_num_voxels_, 0, sizeof(unsigned int), stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(voxelizationLaunch(points, points_size,
          params_.min_x_range, params_.max_x_range,
          params_.min_y_range, params_.max_y_range,
          params_.min_z_range, params_.max_z_range,
          params_.pillar_x_size, params_.pillar_y_size, params_.pillar_z_size,
          params_.getGridYSize(), params_.getGridXSize(), params_.feature_num, params_.max_voxels,
          params_.max_points_per_voxel, hash_table_,
    d_voxel_num_, /*d_voxel_features_*/voxels_temp_, d_voxel_indices_,
    d_real_num_voxels_, stream));
    checkCudaErrors(cudaMemcpyAsync(h_real_num_voxels_, d_real_num_voxels_, sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(featureExtractionLaunch(voxels_temp_, d_voxel_num_,
          *h_real_num_voxels_, params_.max_points_per_voxel, params_.feature_num,
    d_voxel_features_, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));
    return 0;
}
```

1. Memory Initialization: The function starts by clearing the flash memory for each run. It uses `cudaMemsetAsync` to set the `hash_table_` and `voxels_temp_` memory regions to 0xFF asynchronously. It also sets the `d_voxel_num_` and `d_real_num_voxels_` memory regions to 0 asynchronously.

2. Voxelization: Next, the function calls `voxelizationLaunch` with the input `points` array and various parameters such as `min_x_range`, `max_x_range`, `pillar_x_size`, `max_voxels`, etc. This function performs voxelization on the input points, generates a hash table to map voxel offsets to voxel IDs, and records the number of points per voxel in `d_voxel_num_`. The result is stored in `voxels_temp_`.

3. Synchronization: After voxelization, the function synchronizes the CUDA stream to ensure that the previous kernel launch and memory operations are completed before proceeding.

4. Feature Extraction: The function then calls `featureExtractionLaunch` with `voxels_temp_` and other related parameters. This function calculates the average of each feature (x, y, z, intensity, t) for each voxel and stores the results in the `d_voxel_features_` memory region using the `d_voxel_num_` information.

5. Final Synchronization: Lastly, the function synchronizes the CUDA stream again to ensure all computations are completed, and then it returns 0 to indicate successful execution.

Overall, this function efficiently processes a large number of points by leveraging the parallel processing power of the GPU, leading to faster voxelization and feature extraction, crucial steps in point cloud processing and 3D data analysis.