# Actor-Critic Algorithm

We already have value function-based methods (DQN) and policy-based methods (REINFORCE), where value function-based methods only learn a value function, while policy-based methods only learn a policy function. So, a natural question is, is there any way to learn both the value function and the policy function? The answer is Actor-Critic. Actor-Critic is an overall architecture that includes a series of algorithms. Currently, many efficient and cutting-edge algorithms belong to the Actor-Critic algorithm. This chapter will introduce one of the simplest Actor-Critic algorithms. It should be clear that the Actor-Critic algorithm is essentially a policy-based algorithm, because the goal of this series of algorithms is to optimize a policy with parameters, but it will additionally learn the value function, thereby helping the policy function learn better.

## Actor-Critic

To recall, in the REINFORCE algorithm, one of the gradients of the objective function is the trajectory reward, which is used to guide the update of the policy gradient. The REINFORCE algorithm uses the Monte Carlo method to estimate $Q(s, a)$. Can you consider using a value function to replace the $Q$ value in the policy gradient? This is what the Actor-Critic algorithm does. In policy gradient, the gradient can be written in the following general form:

$$
g = \mathbb{E} \left[ \sum_{t=0}^{T} \psi_t \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right]
$$

Among them, $\psi_t$ can have many forms:

1. $\sum_{t'=0}^{T} \gamma^{t'} r_{t'}$: the total return of the trajectory;
2. $\sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$: the reward after action $a_t$;
3. $\sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} - b(s_t)$: baseline improvement;
4. $Q^{\pi_{\theta}}(s_t, a_t)$: action value function;
5. $A^{\pi_{\theta}}(s_t, a_t)$: advantage function;
6. $r_t + \gamma V^{\pi_{\theta}}(s_{t+1}) - V^{\pi_{\theta}}(s_t)$: time series difference residual.

REINFORCE's estimation of the policy gradient through Monte Carlo sampling is unbiased, but the variance is too large. We can use equation (3) to introduce the baseline function $b(s_t)$ to reduce the variance. For example, we can use the Actor-Critic algorithm to estimate the value function $Q$ of a trajectory instead of the return obtained by the Monte Carlo sample, which is equation (4). At this time, we can use the state value function $V$ as the baseline, and subtract this $V$ value from the $Q$ function to get the $A$ function, which we call the advantage function (advantage function). This is equation (5). Furthermore, we can use $Q = r + \gamma V$ to approximate equation (6).Here we mainly consider form (6), that is, the approximate time series differential residual $\psi_t = r_t + \gamma V^{\pi_{\theta}}(s_{t+1}) - V^{\pi_{\theta}}(s_t)$ to guide the policy gradient update calculation. In fact, using $Q$ value or $V$ value is essentially using rewards to guide actions, but using neural networks for estimation can reduce variance and improve robustness. In addition, the REINFORCE algorithm is based on Monte Carlo samples, and the ruler can be updated only after the end of the sequence. This also requires the task to have a clear number of steps, while the Actor-Critic algorithm can be updated after each step and does not make assumptions about the number of steps in the task.

We divide Actor-Critic into two parts, Actor (strategy network) and Critic (value network):

-What the Actor does is interact with the environment and learn a better policy using policy gradient, guided by the Critic value function.
- What Critic needs to do is to learn a value function through the data collected by the interaction between the Actor and the environment. This value function will be used to determine what actions are good and what actions are not good in the current state, and then help the Actor to update its strategy.

The update of Actor adopts the policy gradient principle, so what does Critic know about the update? We denote the critic value network as $V_w$, with parameters $w$. Therefore, we can adopt the temporal difference learning method and define the loss function of the value function for a single data:

$$
\mathcal{L}(w) = \frac{1}{2} \left( r + \gamma V_w(s_{t+1}) - V_w(s_t) \right)^2
$$

As in DQN, we take an approach similar to the target network and use the above $r + \gamma V_w(s_{t+1})$ as the temporal difference target label, without generating gradients to update the new value function. Therefore, the gradient of the value function is:

$$
\nabla_w \mathcal{L}(w) = - \left( r + \gamma V_w(s_{t+1}) - V_w(s_t) \right) \nabla_w V_w(s_t)
$$

The gradient descent method is then used to update the critic value network parameters $w$.

The specific process of the Actor-Critic algorithm is as follows:

- Initialize strategy network parameters $\theta$, value network parameters $w$
- **for** sequence $e = 1 \to E$ **do**:
  - Use the current policy $\pi_{\theta}$ to sample trajectories $\{s_1, a_1, r_1, s_2, a_2, r_2, \dots\}$
  - Calculate $\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$ for each step of data
  - Update value parameters $w = w + \alpha_w \sum_t \delta_t \nabla_w V_w(s_t)$
  - Update strategy parameters $\theta = \theta + \alpha_{\theta} \sum_t \delta_t \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$
- **end for**## TRPO

Methods based on policy gradient include policy gradient algorithm and Actor-Critic algorithm. Although these methods are simple and intuitive, they may encounter training instability during practical application. Let’s review the method based on policy gradient: parameterize the agent’s strategy, design an objective function to measure the quality of the strategy, and maximize this objective function through the gradient ascent method to make the strategy optimal. Specifically, assuming $\theta$ represents the parameters of the policy $\pi_{\theta}$, defined:

$$
J(\theta) = \mathbb{E}_{s_0} \left[ V^{\pi_{\theta}}(s_0) \right] = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]
$$

The goal of policy gradient based methods is to find:

$$
\theta^* = \arg \max_{\theta} J(\theta)
$$

The policy gradient method mainly updates the policy parameters $\theta$ along the $\nabla_{\theta} J(\theta)$ direction. However, this algorithm has an obvious shortcoming: when the policy network parameters are updated and the parameters are updated along the policy gradient, it is likely that one step will be larger than the last time, and the policy changes will be too large, which will affect the training dynamics.

In response to the above problems, we consider finding a trust region (trust region) during update. When updating the policy in this region, we can obtain a certain policy performance security guarantee. This is the main idea of ​​the trust region policy optimization (TRPO) algorithm. The TRPO algorithm was proposed in 2015. It can theoretically ensure the monotonic performance of policy learning and achieve better results than the policy gradient algorithm in practical applications. This confidence interval is the KL divergence.


## PPO

Optimization goals of TRPO:

$$
\max_{\theta} \mathbb{E}_{s \sim \nu^{\pi_{\theta_k}}} \mathbb{E}_{a \sim \pi_{\theta_k}(\cdot|s)} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a) \right], s.t. \mathbb{E}_{s \sim \nu^{\pi_{\theta_k}}} [D_{KL}(\pi_{\theta_k}(\cdot|s), \pi_{\theta}(\cdot|s))] \leq \delta
$$TRPO uses tight constraint expansion approximation, conjugate gradient, linear search and other methods to directly solve it sequentially. The optimization objectives of PPO are the same as TRPO, but PPO uses some relatively simple methods to solve them. Specifically, PPO uses two forms, one is PPO-penalty, and the other is PPO-truncation. We will explore the two supplementary forms for introduction next.

**PPO-Penalty**

PPO-Penalty uses the Lagrange multiplier method to directly put the limit of KL divergence into the objective function, which becomes an unconstrained optimization problem, and the coefficient of KL divergence is continuously updated during the iterative process. That is:

$$
\arg \max_{\theta} \mathbb{E}_{s \sim \nu^{\pi_{\theta_k}}} \mathbb{E}_{a \sim \pi_{\theta_k}(\cdot|s)} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a) - \beta D_{KL}(\pi_{\theta_k}(\cdot|s), \pi_{\theta}(\cdot|s)) \right]
$$

Let $d_k = D_{KL}^{\nu}(\pi_{\theta_k}, \pi_{\theta})$, the update rule of $\beta$ is as follows:

1. If $d_k < \delta / 1.5$, then $\beta_{k+1} = \beta_k / 2$
2. If $d_k > \delta \times 1.5$, then $\beta_{k+1} = \beta_k \times 2$
3. Otherwise $\beta_{k+1} = \beta_k$

Among them, $\delta$ is a hyperparameter set in advance to limit the difference between the learning strategy and the previous round of strategy.

**PPO-Truncation**

Another form of PPO, PPO-Clip (PPO-Clip), is more direct. It restricts the objective function to ensure that the difference between the new parameters and the old parameters is not too large, that is:

$$
\arg \max_{\theta} \mathbb{E}_{s \sim \nu^{\pi_{\theta_k}}} \mathbb{E}_{a \sim \pi_{\theta_k}(\cdot|s)} \left[ \min \left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a), \operatorname{clip} \left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1 + \epsilon \right) \right) \right]
$$Among them, $\operatorname{clip}(x, l, r) := \max(\min(x, r), l)$, that is, $x$ is limited to $[l, r]$. In the above formula, $\epsilon$ is a hyperparameter, indicating the range of truncation (clip).

If $A^{\pi_{\theta_k}}(s, a) > 0$, it means that the value of this action is higher than average. Maximizing this formula will increase $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}$, but will not let it exceed $1 + \epsilon$. On the other hand, if $A^{\pi_{\theta_k}}(s, a) < 0$, maximizing this will reduce $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}$ but not let it exceed $1 - \epsilon$.

![PPO-Clip](./pics/ppo-cut.png)