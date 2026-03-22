# Flash Attention Learning Essay

Due to work reasons, I need to systematically study the operator library of FA3, so starting from the first generation of FA, I systematically studied the development of the entire Flash Attention.

## Self-Attention backpropagation gradient derivation

### The relationship between gradient definition and trace

Let the scalar loss function be $L$. For the matrix $\mathbf{X}$, the gradient $\mathbf{G}_X$ is often remembered as $dX$ in the deep learning circle, which is very easy to be confused with the differential operator. First prove the relationship between gradient and trace:


1. Suppose $L$ is a scalar function whose argument is a matrix $\mathbf{X}$ of $m \times n$. According to the definition of multivariate calculus, when $\mathbf{X}$ produces a small change $d\mathbf{X}$, the small change (total differential) $dL$ of $L$ is the linear superposition of the partial derivatives of each component:

$$dL = \sum_{i,j}^{} \frac{\partial L}{\partial X_{ij}} dX_{ij}$$


For any two matrices $\mathbf{A}$ and $\mathbf{B}$ with the same dimensions, their **Frobenius Inner Product** is defined as the sum of the products of the corresponding elements. This inner product has an identity property:

$$ \langle \mathbf{A}, \mathbf{B} \rangle_F = \sum_{i,j} A_{ij} B_{ij} = \text{Tr}(\mathbf{A}^\top \mathbf{B})$$


The proof is as follows: Consider the matrix multiplication $\mathbf{C} = \mathbf{A}^\top \mathbf{B}$, whose diagonal elements $C_{jj}$ are:


$$C_{jj} = \sum_{i=1}^{m} (\mathbf{A}^\top)_{ji} B_{ij} = \sum_{i=1}^{m} A_{ij} B_{ij}$$


Then its trace (sum of diagonals) is:

$$\text{Tr}(\mathbf{A}^\top \mathbf{B}) = \sum_{j=1}^{n} C_{jj} = \sum_{j=1}^{n} \sum_{i=1}^{m} A_{ij} B_{ij}$$

Applying this property to the gradient, $\mathbf{A} = \nabla_{\mathbf{X}} L$ and $\mathbf{B} = d\mathbf{X}$ are substituted into the above equation:

$$dL = \sum_{i,j} \frac{\partial L}{\partial X_{ij}} dX_{ij} = \text{Tr}((\nabla_{\mathbf{X}} L)^\top d\mathbf{X})$$At this point, we have:

$$dL = \text{Tr}(\mathbf{G}_X^\top d\mathbf{X})$$


### The nature of the trace

1. Transposition invariance: $$ \text{Tr}(\mathbf{A}) = \text{Tr}(\mathbf{A}^\top) $$

2. Circular shift property: $$ \text{Tr}(\mathbf{ABC}) = \text{Tr}(\mathbf{BCA}) = \text{Tr}(\mathbf{CAB}) $$

3. Linear: $$ d(\text{Tr}(\mathbf{A})) = \text{Tr}(d\mathbf{A}) $$

### Backpropagation gradient

In forward propagation: $O = PV$, where $P \in \mathbb{R}^{N \times N}, V \in \mathbb{R}^{N \times d}, O \in \mathbb{R}^{N \times d}$. Given the output gradient $\mathbf{G}_O$ (i.e. $dO$), find $dV$ (i.e. $\frac{\partial L}{\partial V}$):

First do the partial differential $dO = P(dV)$ (treating $P$ as a constant). Substitute into the definition: $dL = \text{Tr}(\mathbf{G}_O^\top dO) = \text{Tr}(\mathbf{G}_O^\top P dV)$. Using the properties of traces, $dV$ is isolated on the right side: $dL = \text{Tr}((\mathbf{G}_O^\top P) dV)$. Therefore, $\mathbf{G}_V^\top = \mathbf{G}_O^\top P \implies \mathbf{G}_V = P^\top \mathbf{G}_O$. Conclusion: $dV = P^\top dO$.


Likewise, find $dP$ (i.e. $\frac{\partial L}{\partial P}$). First perform partial differentiation: $dO = (dP)V$ (treating $V$ as a constant). Substitute into the definition: $dL = \text{Tr}(\mathbf{G}_O^\top dP V) = \text{Tr}(V \mathbf{G}_O^\top dP)$. The comparison yields $\mathbf{G}_P^\top = V \mathbf{G}_O^\top \implies \mathbf{G}_P = \mathbf{G}_O V^\top$. Conclusion: $dP = dO V^\top$Further derive the attention layer, forward formula: $S = QK^\top$, where $Q, K \in \mathbb{R}^{N \times d}, S \in \mathbb{R}^{N \times N}$. Known conditions: gradient $\mathbf{G}_S$ (that is, $dS$ after Softmax backpropagation), find $dQ$ (that is, $\frac{\partial L}{\partial Q}$):

Partial differential $dS = (dQ)K^\top$. Substitute into the definition: $dL = \text{Tr}(\mathbf{G}_S^\top dQ K^\top) = \text{Tr}(K^\top \mathbf{G}_S^\top dQ)$. Therefore $\mathbf{G}_Q^\top = K^\top \mathbf{G}_S^\top \implies \mathbf{G}_Q = \mathbf{G}_S K$, conclusion $dQ = \mathbf{G}_S K$.

Find $dK$ (i.e. $\frac{\partial L}{\partial K}$). First perform partial differentiation: $dS = Q d(K^\top) = Q(dK)^\top$. Substitute into the definition: $dL = \text{Tr}(\mathbf{G}_S^\top Q (dK)^\top) = \text{Tr}(dK Q^\top G_S)$. Therefore $\mathbf{G}_K^\top = Q^\top \mathbf{G}_S \implies \mathbf{G}_K = \mathbf{G}_S^\top Q$, conclusion: $dK = \mathbf{G}_S^\top Q$.


All conclusions are summarized as follows:

$$
\begin{aligned}
    dV &= P^\top dO \\
    dP &= dO V^\top \\
    dQ &= \mathbf{G}_S K \\
    dK &= \mathbf{G}_S^\top Q
\end{aligned}
$$

### Notation description

In the previous notation, we regarded $dV$ as shorthand for $\frac{\partial L}{\partial V}$, which mathematically confuses differential operators and partial differential notation; but in the field of deep learning, this notation is quite common. In order not to write fractions in mathematical derivation, the most rigorous writing method should be $\frac{\partial L}{\partial \mathbf{O}} \in \mathbb{R}^{N \times d}$, but when writing complex backpropagation algorithms (such as Flash Attention), if the entire article is $\frac{\partial L}{\partial Q}, \frac{\partial L}{\partial K}, \frac{\partial L}{\partial V}$, the formula will become very bloated and difficult to read. Therefore, in the field of deep learning, it is customary to use "d + variable name" to directly represent "the partial derivative matrix of the loss function on this variable". $dO$ is actually the abbreviation of $\frac{\partial L}{\partial O}$. $dV$ is actually the abbreviation of $\frac{\partial L}{\partial V}$. This writing method corresponds to the code implementation: if you write PyTorch’s underlying C++ operators or CUDA kernel functions, you will find that the variable names are chosen like this:

```Python
# O is the output of forward propagation
# grad_output is the gradient returned from the previous layer (i.e. dO)
# grad_V is the gradient we want to calculate (i.e. dV)
def backward(grad_output):
    # According to the formula dV = P.T @ dO
    grad_V = P.t() @ grad_output
    return grad_V
```

We will not define a variable called `partial_L_over_partial_O`, but directly call it `grad_output` or `dO`.