# Policy gradient algorithm

Q-learning, DQN and DQN improved algorithms are all value-based methods. Q-learning is an algorithm that deals with finite states, while DQN can be used to solve continuous state problems. In reinforcement learning, in addition to value function-based methods, there is also a very classic method, which is the policy-based method. Comparing the two, the value function-based method mainly learns the value function, and then derives a strategy based on the value function. There is no explicit strategy in the learning process; while the strategy-based method directly and explicitly learns a target strategy. Policy gradient is the basis of policy-based methods. This chapter starts with the policy gradient algorithm.

## Policy gradient

> Gradient policy method is on policy algorithm.

Policy-based methods first need to parameterize the policy. Assuming that the target policy $\pi_{\theta}$ is a randomized policy, we can use a linear model or neural network to represent such a policy model, input a certain state, and then output a probability distribution of an action. Our goal is to update the expected return of a strategy in the environment. We define the objective function of the gradient algorithm as:

$$ J(\theta) = \mathbb{E}_{s_0}\left[V^{\pi_{\theta}}(s_0)\right] $$

Among them, $s_0$ represents the initial state. Now that we have the objective function, after we obtain the gradient of the objective function against the policy parameter $\theta$, we can use the gradient ascent method to maximize the objective function to obtain a better strategy.

$$
\begin{aligned}
\nabla_{\theta}J(\theta) &\propto \sum_{s \in S} \nu^{\pi_{\theta}}(s) \sum_{a \in A} Q^{\pi_{\theta}}(s, a) \nabla_{\theta} \pi_{\theta}(a|s) \\
&= \sum_{s \in S} \nu^{\pi_{\theta}}(s) \sum_{a \in A} \pi_{\theta}(a|s) Q^{\pi_{\theta}}(s, a) \frac{\nabla_{\theta} \pi_{\theta}(a|s)}{\pi_{\theta}(a|s)} \\
&= \mathbb{E}_{\pi_{\theta}}\left[Q^{\pi_{\theta}}(s, a) \nabla_{\theta} \log \pi_{\theta}(a|s)\right]
\end{aligned}
$$

It should be noted that because the subscript in the expectation in the above formula is $\pi_{\theta}$, the policy gradient algorithm is an online policy (on-policy) algorithm, that is, the data sampled by the current policy $\pi_{\theta}$ must be used to calculate the gradient. If you intuitively understand the formula of policy gradient, you can find that in each state, the modification of the gradient is to allow the policy to sample more actions that bring high $Q$ values ​​and less to sample actions that bring lower $Q$ values.## REINFORCE algorithm

- Initial chemical model parameters $\theta$
- for $e = 1 \to E$:
  1. Use the current strategy $\pi_{\theta}$ to sample trajectories $\{s_1, a_1, r_1, s_2, a_2, r_2, \ldots, s_T, a_T, r_T\}$
  2. Calculate the return at each time point of the current trajectory $\sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ and record it as $\psi_t$
  3. Update $\theta$, $\theta = \theta + \alpha \sum_{t} \psi_t \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$
- end for

The REINFORCE algorithm is a typical representative of policy gradient and even reinforcement learning. The agent directly interacts with the environment according to the current policy, directly calculates the gradient of the policy parameters through the sampled trajectory data, and then updates the current policy to move it closer to the goal of maximizing the expected return of the policy. This learning method is a typical learning from interaction, and its optimization goal (ie, the expected return of the strategy) is exactly the performance of the final strategy used, which is more direct than the optimization goal of the value-based reinforcement learning algorithm (generally the minimization of the temporal difference error). The REINFORCE algorithm can theoretically guarantee local optimality. It actually uses the Monte Carlo method to sample trajectories to estimate the action value. A major advantage of this approach is that it can obtain unbiased gradients. However, precisely because of the use of the Monte Carlo method, the variance of the gradient estimate of the REINFORCE algorithm is very large, which may cause a certain degree of instability. This is also a problem that the Actor-Critic algorithm will be introduced to solve.