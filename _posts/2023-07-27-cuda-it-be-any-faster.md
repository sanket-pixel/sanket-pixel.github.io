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





<!-- ## 2. Inside the Code
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
