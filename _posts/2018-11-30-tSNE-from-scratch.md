---
layout: post
title: "t-SNE from scratch (only using numpy)"
author: Qinkai WU
abstract: t-SNE, a powerful and prevailing technique for dimensionality reduction. Here I'll briefly go through its mechanism and also present python code for implementing it from the very beginning.
id: 2018113001
math: True
---
  
As a powerful dimensionality reduction method, t-SNE has been applied in a wide variety of areas since its first introduction by L.v.d. Maaten and G. Hinton in 2008.
  
The first time I got to know t-SNE was from a biomedical research paper on cancer immunology, which shows all the single cells in a 2D plane with axes labeled *t-SNE 1* and *t-SNE 2*. That left me an impression that t-SNE might be a technique just like PCA but more powerful.
  
Soon afterwards I had to analyze single-cell RNA-seq data by myself and using t-SNE frequently through some easy-to-use R packages with only a few lines of code. Only several months ago I occasionally found that t-SNE is actually based on machine learning and its underlying implementation including many familiar concepts like cross entropy and gradient descent. So instantly I've developed a huge interst on it. To better understand it, now I'm gonna write a t-SNE algorithm from scratch (only using numpy).
  
***
  
## Intro
*[Jump directly to the code](#code)*
  
The primary purpose of dimensionality reduction is to effectively reduce the number of variables while preserve as much information as possible from the original data space. But when the datasets become extremly large and complex, linear method like PCA, trying to find the orthogonal directions that maximize the variance, do not always perform well on finding the best low-dimensional manifolds. Then some other non-linear dimensionality reduction methods like Sammon mapping, LLE and Isomap were introduced. Actually Maaten's paper covers a comparision between them and t-SNE using MNIST as the example, the result being t-SNE performs best.



t-SNE, fully named t-distributed stochastic neighbor embedding, can be considered as a refined version of SNE.

Given a bunch of points with high dimensions, SNE uses Gaussian distribution to model the similarity between every pairwise points. More precisely, it converts the euclidean distance between points into probability that measures how alike the two points are (the larger the distance, the smaller the probability). 

In the high-dimensional space, center a Gaussian over a point $i$, we get:   

$$
{p_ {j \mid i} = \frac{\exp(- \mid  \mid  x_i -x_j  \mid  \mid  ^2 / (2 \sigma^2_i ))} {\sum_{k \neq i} \exp(- \mid  \mid  x_i - x_k  \mid  \mid  ^2 / (2 \sigma^2_i))}} \tag{1}
$$

Also define $p_{i \mid i} = 0$ because there's no need to care the similarity between one and oneself. 
(Note that $\sigma_i$ is the variance of Gaussian when centered $i$ and how to select a reasonable $\sigma_i$ will be discussed later in following paragraphs.) 
  
While for the low-dimensional counterparts, we preset the $\sigma$ to $\frac{1}{\sqrt 2}$, therefore
  
$$
{q_ {j \mid i} = \frac{\exp(- \mid  \mid  y_i -y_j  \mid  \mid  ^2)} {\sum_{k \neq i} \exp(- \mid  \mid  y_i - y_k  \mid  \mid  ^2)}} \tag{2}
$$


Clearly the next step would be finding a way to measure the discrepancy between $p_{j \mid i}$ and $q_ {j \mid i}$ and minimize it so the model in low dimension can best represent its high-dimensional counterpart.
    

The original paper mentioned KL-divergence:

$$
D_{KL}(P_i||Q_i)=\sum_i \sum_j p_{j \mid i} \cdot log \frac{p_{j \mid i}}{q_{j \mid i}} \tag{3}
$$  
  

*Note that $log \frac{p_{j \mid i}}{q_{j \mid i}}$ is asymmetrical, that is, a big penalty when large $p_{j \mid i}$ modeled by small $q_{j \mid i}$, which prompts SNE to preserve local structure.*

It works the same way as cross entropy, since the value of $p_{j \mid i} \cdot log \ p_{j \mid i}$ will not change once modeling of high-dimensional datasets has been done. 
  


Like other machine learning tasks, here we can take KL-divergence as the cost function $C$ and will use gradient descent to minimize it.

So now the question is how to get the differentiation. 
  
$$
\frac{\partial C}{\partial y_i} = -  \frac{\partial \sum_i \sum_j p_{j \mid i} \log {q_{j \mid i}}}{\partial y_i} \tag{4}
$$

Take a closer look at Equation 2 and Equation 4, it's not so different from a softmax - cross entropy loss function ([for which I just wrote a note earlier](/2018/11/02/differentiate-a-cross-entropy-cost-function.html)):

$$
a_i = \frac{e^{z_i}}{\sum_k{e^{z_k}}} \\
C = -\sum_i^n{y_i \ln {a_i}} \\
\frac{\partial C}{\partial z_i} = a_i -y_i
$$


Also considering two circumstances:
  
$y_i$ being the center v.s. $y_i$ not being the center, so
  
$$
\frac{\partial C}{\partial (- \mid  \mid  y_i -y_j  \mid  \mid  ^2)} = 
\sum_j (q_{j \mid i} - p_{j \mid i}) + \sum_j (q_{i \mid j} - p_{i \mid j})
$$

Besides,

$$
\frac{\partial (- \mid  \mid  y_i -y_j  \mid  \mid  ^2)}{\partial y_i} = -2(y_i - y_j)
$$

Therefore,

$$
\frac{\delta C}{\delta y_i} = 2 \sum_j (p_{j \mid i} - q_{j \mid i} + p_{i \mid j} - q_{i \mid j})(y_i - y_j) \tag{5}
$$

To speed up training process and avoid getting stuck on local optima, a momentum term can be added to gradient descent:

$$
v^{t} = \beta v^{t-1} - \eta \nabla y^{t} \\
y^t = y^{t-1} + v^{t}
$$

($t$: number of iteration, $\beta$: momentum)

Last question: How to select a proper $\sigma_i$ ?

SNE uses [perplexity](https://en.wikipedia.org/wiki/Perplexity#Perplexity_of_a_probability_distribution) to caculate a $\sigma$ for each point $i$. 

$$
Perp(p_i) = 2^{H(p_i)} \\
H(p_i) = -\sum_j p_{j \mid i} \log_2 p_{j \mid i}
$$

In this case, perplexity are set as a hyperparameter before training. Though the original paper suggests that
SNE is robust to changes in perplexity, in practice it really has no little impact on the result. [See here for more](https://distill.pub/2016/misread-tsne/).  

Since perplexity increases with $\sigma_i$ monotonically ([here's a plot supporting this idea](#plot1)), so the value of $\sigma_i$ can be quickly computed using binary search.


Now return back to t-SNE!  
It mainly has two improvements:
  
- Replacing conditional probability $p_{j \mid i}$ with joint probability $p_{ij}$  
    
    $$
    p_{ij} = \frac{p_{j \mid i} + p_{i \mid j}}{2n}
    $$
- Using t-distribution (here with 1 dgree of freedom) to model the low-dimensional datapoints.  
    
    $$
    q_{ij} = \frac{(1 +  \mid  \mid y_i -y_j \mid  \mid ^2)^{-1}}{\sum_{k \neq l} (1 +  \mid  \mid y_i -y_j \mid  \mid ^2)^{-1}}
    $$

    t-distribution is heavily tailed so similar points are clustered more closely while dissimilar points are dispersed far apart, as a result, mitigating the crowding problem.  
    ![t_vs_gaussian](/img/blog/2018113001/t_dis.png)

So, the new gradient:
  
$$
\frac{\delta C}{\delta y_i} = 4 \sum_j(p_{ij}-q_{ij})(y_i-y_j)(1+ \mid  \mid y_i-y_j \mid  \mid ^2)^{-1}
$$

## Code
<a name='code'></a>

```python
import numpy as np
import matplotlib.pyplot as plt
```

Since MNIST almost becomes the "Hello world" for any machine learning algorithm, I'll also get started with it.


```python
# randomly pick 2000 points from MNIST
n = 2000

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
np.random.seed(42)
ids = np.random.choice(range(70000),n)
X_train, y_train = X[ids], y[ids]
print(X_train.shape, y_train.shape)
```

    (2000, 784) (2000,)



```python
def pairwise_squared_d(X):
    '''
    X should be a m(instances)*n(features) matrix
    similar to (a-b)^2 = a^2 + b^2 - 2*a*b, 
    pairwise euclidean suqared distance of a matrix can be calculated as follows:
    '''
    sum_X = np.sum(np.square(X), 1, keepdims=True)
    suqared_d = sum_X + sum_X.T - 2*np.dot(X, X.T)
    sd = suqared_d.clip(min=0)
    return sd
```


```python
def cal_p(suqared_d, sigma, i):
    """
    Given a line of squared distance and sigma,
    return probability and perplexity
    """
    a = 2*sigma**2
    prob = np.exp(-suqared_d/a)
    prob[i] = 0
    #print(prob)
    prob = prob/np.sum(prob)    
    H = -np.sum([p*np.log2(p) for p in prob if p!=0])
    perp = 2**H
    #print(H)
    return prob,perp
```


```python
def search_sigma(x, i, PERPLEXITY, tol): 
    """
    tol: tolerable difference between
    specified and calculated value of PERPLEXITY
    """
    # binary search
    sigma_min, sigma_max = 0, np.inf       
    prob,perp = cal_p(x, sigma[i], i)
    perp_diff = PERPLEXITY - perp                 
    times = 0
    hit_upper_limit = False
    while (abs(perp_diff) > tol) and (times<50):
        #print(perp_diff, sigma_min, sigma_max)          
        if perp_diff > 0:
            if hit_upper_limit:
                sigma_min = sigma[i]
                sigma[i] = (sigma_min + sigma_max)/2
            else:
                sigma_min, sigma_max = sigma[i], sigma[i]*2
                sigma[i] = sigma_max
        else:
            sigma_max = sigma[i]
            sigma[i] = (sigma_min + sigma_max) / 2
            hit_upper_limit = True
        prob,perp = cal_p(x, sigma[i], i)
        perp_diff = PERPLEXITY - perp  
        times = times + 1
    #print(times) typically around 20 when tol=1e-4
    return prob
```


```python
def get_prob(X, PERPLEXITY=30, tol=1e-4):
    """
    sending each row of the squared distance matrix to
    search_sigma in turn. 
    Getting the final probability matrix.
    """
    n = X.shape[0]
    squared_d = pairwise_squared_d(X)
    squared_d = squared_d/np.std(squared_d, axis=-1)*10
    # init
    pairwise_prob = np.zeros((n,n))
    global sigma
    sigma = np.ones(n)

    for i in range(n):
        x = squared_d[i]
        prob = search_sigma(x, i, PERPLEXITY, tol)
        pairwise_prob[i] = prob
        if i%100 == 0:
            print("processed %s of total %s points"%(i,n))
    return pairwise_prob
```


```python
def pca(x, n_components=None):
    """
    Usually performing PCA first, then select
    dozens of PCs for running tSNE
    """
    print("Preprocessing the data using PCA...")
    vec, val = np.linalg.eig(np.dot(x.T, x))
    assert np.alltrue(np.imag(val)) == False
    """
    if not specifying n_components, taking out the 
    top i components that account for 80% variances
    """
    if n_components:
        return np.real(np.dot(x, val[:,0:n_components]))
    else:
        v_p = vec/sum(vec)
        v_s, i = 0, 0
        while v_s < 0.8:
            v_s += v_p[i]
            i += 1
        return np.real(np.dot(x, val[:,0:i]))
```


```python
scale = lambda x: np.nan_to_num((x-np.mean(x,axis=0))/np.std(x,axis=0), 0)

P =get_prob(pca(scale(X_train)),30, 1e-2)
```
     

    Preprocessing the data using PCA...
    processed 0 of total 2000 points
    processed 100 of total 2000 points
    processed 200 of total 2000 points
    processed 300 of total 2000 points
    processed 400 of total 2000 points
    processed 500 of total 2000 points
    processed 600 of total 2000 points
    processed 700 of total 2000 points
    processed 800 of total 2000 points
    processed 900 of total 2000 points
    processed 1000 of total 2000 points
    processed 1100 of total 2000 points
    processed 1200 of total 2000 points
    processed 1300 of total 2000 points
    processed 1400 of total 2000 points
    processed 1500 of total 2000 points
    processed 1600 of total 2000 points
    processed 1700 of total 2000 points
    processed 1800 of total 2000 points
    processed 1900 of total 2000 points



```python
assert not np.any(np.isnan(P))

P = P + np.transpose(P)

# strangely division by n resulting worse performance
# so I comment out it
P = P / (2)#*n)
```


```python
from collections import defaultdict

tsne_Di = defaultdict(list)
```


```python
def runTSNE(learning_rate, momentum, no_dim, max_iter):
    key_ = str(learning_rate)+'__'+str(momentum)
    
    # randomly assign initial values to y
    y_ = np.random.normal(loc=0,scale=0.01,size=(n,no_dims))
    li = tsne_Di[key_]
    
    v = 0
    print("Cross entropy:")
    for iter in range(max_iter):
        y_s_dist = pairwise_squared_d(y_)
        q = 1/(1+y_s_dist)
        np.fill_diagonal(q,0)
        Q = q/np.sum(q, axis=1, keepdims=True)
        y_f = y_.flatten()
        d = y_f.reshape(no_dims, n, 1, order='F') - y_f.reshape(no_dims, 1, n, order='F')

        CE = -P* np.log2(Q)
        np.fill_diagonal(CE, 0)
        
        if iter%2==0:
            li.append(y_.copy())
        if iter%10==0:
            print(CE.sum())

        gd = 4*(P-Q)*q*d
        gradient = np.sum(gd, axis=2).T
        
        v = learning_rate*gradient + momentum*v    
        y_ = y_ - v
```


```python
no_dims = 2
max_iter = 200
learning_rate = 0.6
momentum = 0.8
```


```python
runTSNE(learning_rate, momentum, no_dims, max_iter)
```

    Cross entropy:
    43860.24585926429
    34840.78937021288
    31417.71615916952
    30631.356223998468
    30337.61785442229
    30127.894209984413
    29964.32210160135
    29813.351371003973
    29736.49460789757
    29661.16210876926
    29599.542134490308
    29536.485370768798
    29489.902813017707
    29472.38112622571
    29423.83910027206
    29394.045214346737
    29368.16915262193
    29356.938758905628
    29352.560458764772
    29323.893431811342



```python
i =-1
key_ = str(learning_rate)+'__'+str(momentum)
li = tsne_Di[key_]
plt.figure(figsize=(6,5))
plt.scatter(li[i][:,0],li[i][:,1], c=y_train, cmap='tab10', s=2)
plt.colorbar()
```

![png](/img/blog/2018113001/output_18_1.png)

It works! Although not looks perfect. That's because getting a good t-SNE result requires tuning parameters, also I didn't consider many other optimizing tricks.

Make a scatter plot of $y$ at each iteration so we can see how the datapoints in low-dimensional space walk around as gradients reducing.  


```python
import os
os.makedirs('tsne')

ii=0
for i in li:
    plt.figure(figsize=(8,8))
    plt.xlim=()
    plt.scatter(i[:,0], i[:,1], c=y_train, cmap='tab10')
    plt.savefig('tsne/'+str(ii)+'.png')
    plt.close()
    ii+=1
```

Shell command, combining pngs to gif
```shell
convert                                                         \
  -delay 0                                                      \
   $(for i in $(seq 0 1 51); do echo tsne/${i}.png; done)       \
  -loop 0                                                       \
   animated_tsne.gif

```

![animated_tsne](/img/blog/2018113001/animated_tsne.gif)

## Appendix
  
<a name="plot1"></a>
*perplexity increases with $σi$ monotonically*

```python
dis = pairwise_squared_d(pca(scale(X_train)))

sigma = np.linspace(0.5, 100, 1000)

for l in range(10):
    perp = [cal_p(dis[l], i, l)[1] for i in sigma]
    plt.plot(sigma,perp)
plt.xlabel("$\sigma$")
plt.ylabel("Perplexity")
```

![png](/img/blog/2018113001/output_27_2.png)

  
***
Actually, `scikit-learn` already provide a [out-of-the-box class for t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html):
  
```python
from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2).fit_transform(X)
```
  
**Running t-SNE can be fairly slow, implementing it using GPU might be a good idea.**

## Reference  
  
1. [t-SNE – Laurens van der Maaten](https://lvdmaaten.github.io/tsne/)
2. [Visualizing Data using t-SNE - Journal of Machine Learning Research](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
4. [Visualizing Data Using t-SNE, GoogleTechTalk, June 24, 2013](https://www.youtube.com/watch?v=RJVL80Gg3lA&t=2706s)
