# Timing difference algorithm

The dynamic programming algorithm requires that the Markov decision process is known, that is, the environment in which the agent interacts is completely known. The agent does not need to actually interact with the environment to sample data. It can directly use the dynamic programming algorithm to solve the optimal value or strategy. This is just like for supervised learning tasks. If the distribution formula of the data is directly and explicitly given, then the model parameters can be updated by directly minimizing the generalization error of the model at the desired level without sampling any data points.

But this is not realistic in most scenarios. The main methods of machine learning update the model for specific data points when the data distribution is unknown. Even for most realistic scenarios of reinforcement learning, the state transition probability of the Markov decision process cannot be written, let alone dynamic programming. In this case, the agent can only interact with the environment and learn through sampled data. This type of learning method is collectively called model-free reinforcement learning.

Different from dynamic programming algorithms, model-free reinforcement learning algorithms do not need to know the reward function and state transition function of the environment in advance, but directly use the data sampled during the interaction with the environment to learn. This part will explain two classic algorithms in model-free reinforcement learning: Sarsa and Q-learning, both of which are reinforcement learning algorithms based on temporal difference (TD). At the same time, this chapter will also introduce a set of concepts: online policy learning and offline policy learning. Generally speaking, online policy learning requires the use of samples sampled under the current policy for learning. Once the policy is updated, the current sample is abandoned; while offline policy learning can repeatedly use the collected experience, can better utilize historical data, and has smaller sample complexity (the number of samples that need to be sampled in the environment to achieve convergence results of the algorithm), which makes it more widely used.

## Timing difference algorithm

**TD (Temporal Difference) is a method used to estimate the value function of a strategy, which combines the ideas of Monte Carlo and dynamic programming algorithms. The similarity between the temporal difference method and Monte Carlo is that it can learn from sample data without knowing the environment in advance; the similarity with dynamic programming is that based on the idea of ​​the Bellman equation, the value estimate of the subsequent state is used to update the value estimate of the current state. ** Let’s review how the Monte Carlo method incrementally updates the value function in the multi-armed bandit section:

- TD learns directly from episodes of experience without knowing the model
- TD uses the Bellman equation to update the value function
- TD uses step-by-step bootstrapping to learn without knowing the complete episode

$$V(s_t) \leftarrow V(s_t) + \alpha [G_t - V(s_t)]$$

Here we replace $\frac{1}{N(s)}$ with $\alpha$, which represents the step size for updating the value estimate. $\alpha$ can be taken as a constant, in which case the update step size is fixed; $\alpha$ can also be taken as a decay sequence (such as $\frac{1}{t}$), in which case the update step size will decrease as time increases. Monte Carlo methods must wait until the end of a sequence to calculate a single return $G_t$, while temporal difference methods only rely on value estimates of the current and subsequent states without waiting for the entire sequence to end. A variant of the temporal difference algorithm is the incremental update method of prediction error based on the Bellman equation:$$V(s_t) \leftarrow V(s_t) + \alpha [R_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

Among them, $R_{t+1} + \gamma V(s_{t+1})-V(s_t)$ is the target value error (error) of the current strategy for temporal difference. The temporal difference algorithm uses the product of the next step length as the update amount of the state value.

## Sarsa Algorithm

Use a greedy algorithm to select actions based on action values to interact with the environment, and then update the current state action value based on the rewards obtained and subsequent state value estimates. The formula is as follows:

$$
\pi(a|s) = \begin{cases}
\frac{\epsilon/|A| + 1 - \epsilon}{1/|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
1/|A| & \text{Other actions}
\end{cases}
$$

Now, we can get an actual reinforcement learning algorithm based on the temporal difference method. This algorithm is called Sarsa because its action value update uses the current state s, the current action a, the reward r obtained, the next state s' and the next action a'. After splicing these symbols, the name Sarsa is obtained. The following is the specific algorithm of Sarsa as follows:

- Initialize $Q(s, a)$
- for sequence e = 1 -> $E$ do:
  - Get the initial state s
  - Use $\epsilon$-greedy strategy to select action a in current state s based on Q
    - for time step t = 1 -> $T$ do:
      1. Get r, s' of environmental feedback
      2. Use the $\epsilon$-greedy strategy to select action a' in the current state s' according to Q.
      3. $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$
      4. $s \leftarrow s', a \leftarrow a'$
    - end for
  - end for
- end for

As a result, the Sarsa algorithm updates the policy value function formally, and the final policy is updated indirectly as described above:

$$
\pi(a|s) = \begin{cases}
\frac{\epsilon/|A| + 1 - \epsilon}{1/|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
1/|A| & \text{Other actions}
\end{cases}
$$Monte Carlo methods utilize the reward at each step after the current state without using any estimated value function. The sequential difference algorithm only uses the reward of one step and the reward of the next step without waiting for the end of a sequence. The value function of the Monte Carlo algorithm is unbiased, but has a relatively large variance, because the future transfer of each step may have an uncertain transfer direction; the sequential difference algorithm has a smaller variance, but because it only focuses on one transfer and uses the value estimate of the next state to replace subsequent value estimates, TD is biased and no real value is used in its value estimate. ** So is there any way to combine the advantages of the two? The idea of multi-step timing difference is to update in a compromise way:

$$G_t = r_t + \gamma Q(s_{t+1}, a_{t+1})$$

Replace with:

$$G_t = r_t + \gamma r_{t+1} + \cdots + \gamma^n Q(s_{t+n}, a_{t+n})$$

So:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

Replace with:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma r_{t+1} + \cdots + \gamma^n Q(s_{t+n}, a_{t+n}) - Q(s_t, a_t)]$$

## Q-Learning

In addition to Sarsa, there is also a very famous reinforcement learning algorithm based on the temporal difference algorithm - Q-learning. The biggest difference between Q-Learning and Sarsa is the temporal difference update method of Q-learning:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

The specific process of the Q-learning algorithm is as follows:

- Initialize $Q(s, a)$
- for sequence e = 1 -> $E$ do:
  - Get the initial state s
  - for time step t = 1 -> $T$ do:
    1. Use the $\epsilon$-greedy strategy to select action a in the current state s based on Q.
    2. Get r, s' of environmental feedback
    3. $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
    4. $s \leftarrow s'$
  - end for
- end forWe can use the idea of value iteration to understand Q-learning, that is, Q-learning is directly estimating Q*, because the Seemann optimal equation of the action value function is:

$$Q^*(s, a) = r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) \max_{a'} Q^*(s', a')$$

While Sarsa estimates the action value function of the current $\epsilon$-greedy strategy, it should be emphasized that the update of Q-learning does not have to use the action obtained by the current greedy strategy $argmax_a Q(s, a)$, but can directly update the state action value according to the update formula without relying on the selection of the current strategy.

The core point of the update is that we use a $\epsilon$-greedy strategy to interact with the environment, and then update the current state action value based on the reward obtained and the action value function of the next state. Sarsa is an on-policy algorithm that must use the current $\epsilon$-greedy strategy to select actions, while Q-learning is an off-policy algorithm. Both concepts are very important in reinforcement learning.

1. In Q learning, both behavior and target policies will improve; target policy is a greedy strategy, and behavior policy can be a random strategy or a $\epsilon$-greedy strategy;
2. The sarsa algorithm will truly choose $A_{t+1}$, while the Q learning algorithm will choose $argmax_a Q(s_{t+1}, a')$ as $A_{t+1}$, so the former is an on-policy algorithm and the latter is an off-policy algorithm;

These two sentences are explained very well by the picture below:

 
<div align="center">
  <img src="./pics/sarsa-q-learning.png" width="50%" />
</div>