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


