---
layout: post
comments: true
title:  "Gradient Descent vs Coordinate Descent"
excerpt: "You may have used coordinate descent for optmizing lots of ML algorithms like Lasso, L2 Norm SVM etc. In this blog we will compare coordinate descent with Gradient descent."
date:   2019-03-06 23:43:00
mathjax: true
---

Gradient descent is most widely used algorithm for finding the local minima which is usually the global minima for convex functions. But in certain cases it is very expensive to compute the derivative than computing the function value itself. In such cases Coordinate descent turns out to be a wonderful algorithm where we take steps along one of the coordinate lines to reach global optimum. However, it is to be kept in mind that gradient descent and coordinate descent usually never converges at precise value and some tolerance must be kept. 

Let us look what these algorithms have to say.

**Algorithm for Gradient descent**
<br>Suppose J is the cost function which we are trying to minimise then 
<br>Repeat unti convergence {
<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta) $$ ( for every j)
<br><br>}



**Algorithm for Coordinate descent**
<br>Consider an unconstrained optimization problem $$min_{\alpha} W(\alpha_1, \alpha_2,..., \alpha_m)$$. Think of W as some function of parameters $$ \alpha_i$$'s.
<br>Loop until convergence: {
	<br>
	&nbsp; &nbsp; &nbsp; For $$ i=1,...,m $$ {
	<br>
	&nbsp; &nbsp; &nbsp; $$ \alpha_i := arg min_{\hat{\alpha_i}} W(\alpha_1, ..., \alpha_{i-1},\hat{\alpha_i},\alpha_{i+1}, ..., \alpha_m) $$	
	<br>
	&nbsp; &nbsp; &nbsp;}

}

<br>
In the inner loop we try to optimize W with some fixed $$\alpha_i$$ and hold all other parameters except $$\alpha_i$$. This method is very efficient when we are able to compute <i>arg min</i> efficiently.

Some little change in Coordinate descent equips us with Coordinate ascent which is often applicable in L2 SVM where we need to a solve a non-convex problem (NP hard) using convex approximation. 

**L2 SVM Optimisation Problem**
<br>We consider the dual of the original L2 SVM Problem, which is given by:

$$max_{\alpha} W(\alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i,j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle $$
<br>
s.t. &nbsp; $$0\leq\alpha_i \leq C, i = 1,2,.., m $$
<br>
$$\sum_{i=1}^m \alpha_iy^{(i)} = 0$$

 This problem can't be solved directly using coordinate descent as we need to update only one alpha and this wont allow us to keep $$\sum_{i=1}^m \alpha_iy^{(i)} = 0$$. This problem can be solved ingeniously by updating two $$\alpha_i$$'s at a time. This forms the basis for Sequential Minimiation Optimisation (SMO) algorithm for Solving this convex optimization problem. SMO algorithm goes as follows:

1. Select $$\alpha_i, \alpha_j$$ (by some heuristics)
2. Hold all $$\alpha_k$$'s fixed except $$\alpha_i, \alpha_j$$
3. Optimize $$W(\alpha)$$ with respect to $$\alpha_i, \alpha_j$$

<b>Note:</b> By gradient descent we have referred to batch gradient descent in this post.

<b>References:</b>
* [Stanford's CS229 notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes3.pdf)
* [Wikipedia - Coordinate Descent](https://en.wikipedia.org/wiki/Coordinate_descent)
* [John Platt's paper on SMO](https://pdfs.semanticscholar.org/59ee/e096b49d66f39891eb88a6c84cc89acba12d.pdf)  