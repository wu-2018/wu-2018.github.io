---
layout: post
author: Qinkai WU
abstract: Covering all the details on how to calculate the gradient when using softmax and cross entropy in a neural network classifier.
id: 2018110201
math: True
tags: ['Deep Learning', 'Classification']
---

Softmax and cross entropy are widely used for classification tasks.
  
Look at each node from the last softmax layer of a typical neural network classifier:
  
![img1](/img/blog/2018110201/img1.svg)
  
Apparently,
  
$$
z_i=\sum_l w_{il}x_{l}+b_i \\
a_i = \frac{e^{z_i}}{\sum_k{e^{z_k}}}
$$

$a_i$ denotes the $i$-th output node.
  
When using cross entropy as the loss function:

$$
L = -\sum_i^n{y_i \ln {a_i}}
$$

Note here $y$(true) and $a$(predicted) are both a vector, e.g.,

$y = [0,1,0], \quad a = [0.1, 0.6, 0.3]$

Hence, $i$ represents the index of the features and $n$ is the number of features.
  

Since
  
$$
\frac{\partial L }{\partial w_{il}} = \frac{\partial L }{\partial z_i} \cdot \frac{\partial z_i }{\partial w_{il}} \\
\frac{\partial L }{\partial b_i} = \frac{\partial L }{\partial z_i} \cdot \frac{\partial z_i }{\partial b_i}
$$

Also,

$$
\frac{\partial z_i }{\partial w_{il}} = \frac{\partial L }{\partial z_i} \cdot x_l  \\
\frac{\partial z_i }{\partial b_i} = 1 
$$

Now the key point is to get $\frac{\partial L }{\partial z_i}$ :
  
(Consider the specialty of softmax: each $z_i$ will be affected by all the $a_j$ when  calculating partial derivative)
  
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial z_i} &= \frac{\partial L}{\partial a_j} \cdot  \frac{\partial a_j}{\partial z_i} \\
&= \frac{\partial (-\sum_j^n{y_j \ln {a_j}})}{\partial a_j} \cdot  \frac{\partial a_j}{\partial z_i} \\
&=-\sum_j^n{y_j \frac{1}{a_j}} \cdot  \frac{\partial a_j}{\partial z_i}
\end{aligned}
\end{equation}
$$

When $j=i$,
  
$$
\begin{equation}
\begin{aligned}
\frac{\partial a_j}{\partial z_i} &= \frac{\partial ( \frac{e^{z_i}}{\sum_k{e^{z_k}}})} {\partial z_i} \\ &= \frac{e^{z_i} \cdot \sum_k{e^{z_k} - (e^{z_i})^2}}{(\sum_k{e^{z_k} })^2 } \\ &= ( \frac{e^{z_i}}{\sum_k{e^{z_k}}})(1 - \frac{e^{z_i}}{\sum_k{e^{z_k}}}) \\ &= a_i(1-a_i)
\end{aligned}
\end{equation}
$$

  
When $j \neq i$,
  
$$
\frac{\partial a_j}{\partial z_i} = \frac{\partial ( \frac{e^{z_j}}{\sum_k{e^{z_k}}})}
{\partial z_i} = \frac{0 - e^{z_j} e^{z_i}} {(\sum_k{e^{z_k} })^2 } = -a_j a_i
$$


Combine the above two conditions:
  
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial z_i} &= -\sum_j^n{y_j \frac{1}{a_j}} \cdot  \frac{\partial a_j}{\partial z_i} \\ &= - y_i \frac{1}{a_i} \cdot a_i (1-a_i) - \sum_{j \neq i} y_j \frac{1}{a_j} \cdot (-a_j a_i) \\ &= - y_i + a_i y_i + \sum_{j \neq i} y_j a_i \\ &=- y_i + \sum_{j} y_j a_i 
\end{aligned}
\end{equation}
$$

Since $y_i$ is a one-hot vector, therefore 
  
$$
\sum_{j} y_j a_i = a_i \\
\frac{\partial L}{\partial z_i} = a_i -y_i
$$

Now we can see the obvious strength of using cross entropy as cost function - It's so easy to compute the gradient!
