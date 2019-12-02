---
layout: post
comments: true
title:  "Using SPSA for optimizing Neural Networks"
excerpt: "Being a backprop ninja aids you with a powerful tool for training Neural Networks. What if the loss function is non-differentiable? or very expensive to compute? SPSA comes handy in such cases. Let's dive deeper in SPSA in this post."
date:   2019-06-21 20:57:00
mathjax: true
---
Neural Networks are at the core of deep learning. But these are often constrained by Back-propagation algorithm which requires the derivative of Loss function with respect to network parameters. In this post, I will show that Neural Networks are not limited by back-propagation and we can use Simultaneous Perturbation using Stochastic Approximation(SPSA) to find noisy gradients. This technique is highly useful when it is very expensive to compute the gradients of the loss function or it is not differentiable at all. Gradient Descent or any other popular optimisation algorithms like Adam/RMSProp requires to compute Gradient.

### Simultaneous Perturbation for Stochastic Perturbation (SPSA)
Let us formulate our optimisation problem as $$\hat{x} = argmin_x f(x)$$.
Gradient descent is a well known optimisation algorithm for finding the optimum in convex functions. It takes steps as $$ x_{t+1} = x_t - \epsilon \nabla f(x) $$. Suppose $$ g(x) = \nabla f(x) $$
Using First principle of Calculus, this can be computed as 
$$ g(x) = \displaystyle{\lim_{\epsilon \rightarrow 0}}\frac{f(x+\epsilon ) - f(x-\epsilon )}{2\epsilon } $$.
Now SPSA takes the advantage of above equation and uses approximated gradient. We are perturbing $$x_t$$ with some perturbation $$\delta$$ as $$ \hat{g}(x) = \frac{f(x+\delta) - f(x-\delta)}{2\delta} $$ and descend in the direction of above approximated gradient. These gradients are no doubt quite noisy, but as $$ \displaystyle{\lim_{t \rightarrow \infty}} x_t = \hat{x} $$.


>The reader must keep in the mind that backpropagation is a bottom to up approach whereas there is no such constraint on SPSA to update the network parameters. So we often use top to bottom approach while training using SPSA.


#### 1) Pseudo Code for SPSA Update

> $$x_0 \in \mathbb{R}^{m} $$ <br/> 
> $$\forall \, t = 1,.., t_{max} $$ <br/>
> $$\quad$$ set $$\quad a_t = \frac{a}{t^\alpha} $$ <br/>
> $$\quad$$ set $$\quad c_t = \frac{c}{t^\gamma} $$ <br/>
> $$\quad$$ randomly sample $$\quad\delta \sim \mathcal{U}(\{−1, +1\}^m) $$ <br/>
> $$\quad$$ set $$\quad x^+ = x_t + c_t δ $$ <br/>
> $$\quad$$ set $$\quad x^− = x_t − c_t δ $$ <br/>
> $$\quad$$ compute $$\quad \hat{g}(x_t) \quad using $$ <br/>
> $$\quad \quad \hat{g}(x_t)= \frac{f(x^+) − f(x^-) }{2 c_t δ} $$ <br/>
> $$\quad$$ update <br/>
> $$\quad \quad x_{t+1} = x_t − a_t \,\hat{g}(x_t)$$ <br/>


The above algorithm requires **careful** initialisation of $$a_t$$ and $b_t$. $$\delta$$ is initialised using the [Rademacher distribution](https://en.wikipedia.org/wiki/Rademacher_distribution). Now you might be thinking that why it is not being commonly used, One of the main reasons behind that is the slow convergence of SPSA. It requires large number of iterations to converge. Let us look how it can be used for training a neural network. Suppose $$\mathcal{W} = \left \{ W^1, W^2, ...,W^l \right \}$$ be the weights of $$l$$ layers. We are not considering biases for simplicity. Suppose $$W^l_{e}$$ denotes the weights of layer l and epoch e. There is a trade off here between accuracy and computational difficulty. Various other gradient free methods also exists but they are not as efficient and popular as SPSA in deep learning literature.
### 2) Pseudo Code for SPSA Update

>initialize $$W^1_0 ,..., W^l_0$$ <br/>
> // perform $$e_{max}$$ training epochs <br/>
>for all $$ e = 1, ..., e_{max} $$ <br/>
> &nbsp;&nbsp;&nbsp; // iterate over all layers from bottom to top<br/>
> &nbsp;&nbsp;&nbsp; for all $$l = 1, ..., l$$ <br/>
> &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; // keeping matrices $$W^{k\neq l}$$ fixed, use SPSA to update matrix W^l <br/>
> &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; // the objective function to be evaluated is E D, W we defined above<br/>
> &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;$$W^l_{e+1} = SPSA(\mathcal{L}, W^l_e)$$ <br/>

### 3) Python Code
Complete code for toy dataset using numpy only in python can be found at [https://github.com/anshul3899/SPSA-Net](https://github.com/anshul3899/SPSA-Net)
```python
def train(self, inputs, targets, num_epochs, t_max= 200): 
        # initialize layer weights
        weights = []
        np.random.seed(1)
        weights.append(np.random.randn(self.input_dim, self.output_dim))
        # weights.append(np.random.randn(self.latent_dim, self.output_dim))
        # self.W.append(np.random.randn(latent_dim, latent_dim))
        # self.W.append(np.random.randn(latent_dim, output_dim))

        for epoch in range(num_epochs):
            for l in range(len(weights)):
                W_p = np.copy(weights)
                W_m = np.copy(weights)
                for t in range(1,t_max):
                    a_t = self.a / t**self.alpha
                    c_t = self.c / t**self.gamma
                    delta = np.random.binomial(1, p=0.5, size=(weights[l].shape)) * 2. - 1
                    # perturb weights in plus directions
                    W_p[l] = W_p[l] + c_t * delta
                    # compute predictions according to W_p and then compute loss using perturbed weight
                    preds= self.forward(inputs, W_p)
                    loss_p = self.loss( preds, targets)
                    # perturb weights in minus directions
                    W_m[l] = W_m[l] - c_t * delta
                    # compute predictions according to W_m and then compute loss using perturbed weight
                    preds= self.forward(inputs,W_m)
                    loss_m = self.loss( preds, targets)
                    # Compute approximation of the gradient
                    g_hat = (loss_p - loss_m) / (2 * c_t * delta)
                    if loss_m - loss_p != 0 :
                        print("gotcha", loss_m - loss_p, 0.01*np.mean(g_hat))
                    # print("a_t * g_hat mean is: ", np.mean(a_t * g_hat))
                    weights[l] = weights[l] - a_t * g_hat
            if(epoch % 1 == 0):
                preds = self.forward(inputs, weights)
                print("RMSE Loss is: ", np.sum((self.forward(inputs, weights) - targets)**2))
        return weights

    def loss(self, preds, targets):
        return np.sum((preds-targets)**2)
```
**References:**
[Slides of Christian Bauckhage on using SPSA for solving Neural Networks](https://www.researchgate.net/profile/Christian_Bauckhage/project/lectures-on-pattern-recognition/attachment/5a5c9cf14cde266d58831886/AS:583039802462208@1516018929842/download/lecture-18-add-on-2.pdf?context=ProjectUpdatesLog)
