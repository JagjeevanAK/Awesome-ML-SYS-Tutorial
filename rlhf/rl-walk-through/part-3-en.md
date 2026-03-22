# Dynamic programming algorithm

I have been learning dynamic programming algorithms since I was a freshman, but I never understood why there was a proper term "value function" in it at that time. Only now I discovered that it is actually a term inherited from RL. The world line suddenly ended.

This part describes the use of dynamic programming method to solve the optimal strategy of MDP decision-making process. There are two main types of reinforcement learning algorithms based on dynamic programming: one is policy iteration, and the other is value iteration. Among them, policy iteration consists of two parts: policy evaluation and policy improvement. Specifically, the policy evaluation in policy iteration uses the Bellman expectation equation to obtain the state value function of a policy, making it rise steadily. This is a dynamic programming process; while the value iteration directly uses the Bellman optimal equation to perform dynamic programming to obtain the final optimal state value.

These two reinforcement learning algorithms based on dynamic programming require knowing the state transition function and reward function of the environment in advance, that is, the entire Markov decision process needs to be known. In such a white-box environment, there is no need to learn through a large amount of interaction between the agent and the environment, and dynamic programming can be used directly to solve the state value function. However, there are very few white-box environments in reality, which is also the limitation of dynamic programming algorithms. We cannot apply them to many practical scenarios. In addition, policy iteration and value iteration are usually only applicable to finite Markov decision processes, that is, the state space and action space are discrete and limited.

## Strategy iteration algorithm

Strategy iteration is a process in which strategy evaluation and strategy improvement are continuously alternated until the optimal strategy is finally obtained.

### Strategy Evaluation

As mentioned in the section, the exact solution of the policy's state value function is a $n^3$ brute force process, so we consider using the Bellman expectation equation to calculate the numerical solution:

$$ V^{\pi}(s) = \sum_{a \in A} \pi(a|s) \left( r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^{\pi}(s') \right) $$

where $\pi(a|s)$ is the probability that the policy takes action $a$ in state $s$. It can be seen that when we know the reward function and the state transition function, we can calculate the value of the current state based on the value function of the next state. Based on the idea of ​​dynamic programming, calculating the value of the next possible state can be regarded as a sub-problem, and calculating the value of the current state can be regarded as the current problem. Once the solutions to the subproblems are known, the current problem can be solved. More generally, in round 0, we directly specify values ​​for the value functions of all states, and then we can update the value functions of all states in the next round. In this way, it becomes to use the state value function of the previous round to calculate the state value function of the current round, that is:

$$ V^{k+1}(s) = \sum_{a \in A} \pi(a|s) \left( r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^{k}(s') \right) $$According to the Bellman expectation equation, we can know that $V^k = V^{\pi}$ is a fixed point of the above update formula. In fact, it can be shown that the sequence $\{V^k\}$ converges to $V^{\pi}$ when $k \to \infty$. Therefore, the state value diagram function of a strategy can be calculated iteratively. It can be seen that due to the need to continuously update Bellman expectations, state evaluation will actually consume a lot of computational value. In actual practice, if it is a round where the value of $\max_{s \in S} |V^{k+1}(s) - V^k(s)|$ is very small, the evaluation can be ended early. This approach speeds up the convergence and results in a value that is very close to the true value.

### Strategy improvement

We can directly greedily select the action with the largest state value function in each state, that is:

$$
\pi'(s) = \arg \max_{a} Q^{\pi}(s, a) = \arg \max_{a} \{r(s, a) + \gamma \sum_{s'} P(s'|s, a) V^{\pi}(s')\}
$$

This must be the best strategy based on the current strategy evaluation. Of course, our strategy evaluation may be inaccurate, so in the next round of iteration, there will be a new best strategy. The two together form a strategy iteration algorithm that continuously cycles through strategy evaluation and strategy improvement.

## Value iteration algorithm

Different from the policy iteration algorithm, based on the Bellman optimal equation, we can directly iterate to obtain the final value, thereby obtaining the optimal strategy in one step.

$$ V^*(s) = \max_{a \in A} \left\{ r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^*(s') \right\} $$

The way to write it as an iterative update is:

$$ V^{k+1}(s) = \max_{a \in A} \left\{ r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^k(s') \right\} $$

Value iteration is carried out step by step according to the above update method. When $V^{k+1}$ and $V^k$ are the same, it is the fixed point of Bellman's optimal equation, which corresponds to the optimal state value function $V^*$. Then we use:

$$ \pi(s) = \arg \max_{a} \left\{ r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^{k+1}(s') \right\} $$

Just choose the optimal strategy from them. The calculation is naturally much simpler. After iterating multiple rounds of values, the best strategy can be obtained in the last round.