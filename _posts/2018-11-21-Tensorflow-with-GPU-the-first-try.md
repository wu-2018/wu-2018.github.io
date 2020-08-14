---
layout: post
title: "Tensorflow with GPU: the first try"
author: Qinkai Wu
abstract: Installing Tensorflow-GPU 1.12.0 on Ubuntu 18.04. It's also my first time running tensorflow on my own graphics card.
pid: 2018112101
tags: ['Tensorflow', 'Deep Learning']
categories: journal
---
  
Though I've been exploring machine learning for a while and I did understand that GPU can substantially boost the training speed due to its innate architecture, only recently I've made the decision to get myself a graphic card. In large part, it's due to the unbearable slowness caused by the increasing dataset size that I've got to tackle, but more importantly, it's the Double 11's promotion reached a special consensus with my balance.  

![bqb](/assets/img/post/{{ page.pid }}/i0.png)
  
It's only a GeForce GTX960 4G card, but for personal study instead of production-level tasks, it's already capable enough.  

![gtx960](/assets/img/post/{{ page.pid }}/igc.jpg)

Quickly I opened the PC case and everything seems fine, the PCIe slot there has been unoccupied for so long!
  
![pcie](/assets/img/post/{{ page.pid }}/i2.jpg)

But awkawardly, this obsolete computer in my office doesn't even have a 6pin power cable! So I bought an extra SATA 15pin to 6pin PCIe power cable, and it works well.
  
![15-6 pin](/assets/img/post/{{ page.pid }}/i1.jpg)

Next steps should all be related to the software side.  
The drivers are available in Ubuntu's **Software & Updates**, also it can be installed via command line.
  
![driver](/assets/img/post/{{ page.pid }}/i4.png)

To check if the driver works, type `nvidia-smi`, however I didn't get the expected message.  
  
![drive_error](/assets/img/post/{{ page.pid }}/i5.png)
  
But a simple reboot fixed it!  

![drive_ok](/assets/img/post/{{ page.pid }}/i6.jpg)
  
Following the instruction at [Tensorflow's official website](https://www.tensorflow.org/install/gpu), I got [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) from Nvidia.  

After having downloaded these huge files(>2Gb in total), suddenly I found CUDA's version - 10.0,  was still not specifically supported by tensorflow. 
  
![cuda_v](/assets/img/post/{{ page.pid }}/i8.png)

So I tried to seek for CUDA Toolkit v9.0 on nvidia's website but found nothing until google helped me get its link.  

Another unexpectation - seemingly this version was not made for 18.04...  

![cuda](/assets/img/post/{{ page.pid }}/i7.png)

Maybe this nunace is totally negligible, but who knows what potential mess would happen later on?
  
Then I noticed another trouble: in order to get CUDA 9.0 work, I have to degrade GCC to version 6 ...  

All these nuisance collectively provoked me to find a second solution - that is, building tensorflow from source.

Thanks to some active online communities, I've also tried several unofficial `.whl` file built under the environment Ubuntu18.04 + CUDA10.0, but disappointingly, when I typed `import tensorflow`, all of them ended up giving error messages or crashing rapidly. 
  
Then I realized, "building it myself" should be the most straightforward solution. Fortunately I found an extremely helpful instruction [here](https://www.python36.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/comment-page-2/#comments)!

After finishing `./configure`, upon using bazel to build Tensorflow, another error happend:
  
![bazel_e](/assets/img/post/{{ page.pid }}/i9.png)

Then I found solution [here at github](https://github.com/tensorflow/tensorflow/issues/23401). Simply add a new line `import /home/wqk/A/tensorflow/tools/bazel.rc` to `.tf_configure.bazelrc` then run bazel again.
  
**Note that adding `--local_resources=RAM,CPU,I/O` e.g., `--local_resources=2048,1,1.0` after `bazel build` would help, since bazel can easily consume a lot of resource and even cause computer stuck.**

From now on, just wait. It'll take a huge amount of time, thinking hours, but of course it depends on the hardwares'performance and may varies from computer to computer. In my case, I just left my computer alone and the next morning when I came back to office, everything's done. 

![finish_build](/assets/img/post/{{ page.pid }}/i10.png)

The last step:  
Use `bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg` to generate the `.whl` file, then `cd tensorflow_pkg && sudo pip3 install *.whl`!


See if tensorflow can work properly:


```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```




    [name: "/device:CPU:0"
     device_type: "CPU"
     memory_limit: 268435456
     locality {
     }
     incarnation: 17282486505901076303, name: "/device:XLA_CPU:0"
     device_type: "XLA_CPU"
     memory_limit: 17179869184
     locality {
     }
     incarnation: 17524073765003166309
     physical_device_desc: "device: XLA_CPU device", name: "/device:XLA_GPU:0"
     device_type: "XLA_GPU"
     memory_limit: 17179869184
     locality {
     }
     incarnation: 1434134268135013825
     physical_device_desc: "device: XLA_GPU device", name: "/device:GPU:0"
     device_type: "GPU"
     memory_limit: 3491299328
     locality {
       bus_id: 1
       links {
       }
     }
     incarnation: 3003047154314331382
     physical_device_desc: "device: 0, name: GeForce GTX 960, pci bus id: 0000:01:00.0, compute capability: 5.2"]




```python
import tensorflow as tf

a = tf.constant([1,2,3,4], shape=[2,2], name='a')
b = tf.constant([1,2,3,4,5,6,7,8], shape=[2,4], name='b')
c = tf.matmul(a,b)

sess = tf.Session()
print(sess.run(c))
sess.close()
```

    [[11 14 17 20]
     [23 30 37 44]]




