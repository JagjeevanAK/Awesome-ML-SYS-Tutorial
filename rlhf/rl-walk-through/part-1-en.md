# Reinforcement Learning - Multi-Armed Bandit

These are the RL course notes from UCLA in the early spring of 2025. Thanks to Teacher Zhou Bolei for the courseware. I named this series Demystifying Reinforcement Learning because I have always wanted to do some work on the RLHF framework, so it is very important to be intuitive about RL algorithms. Due to my personal learning habits, I am not interested in theory. Fortunately, for ML SYS researchers, such intuition can already bring huge help.

In a trade-off between intuition and solid theory, some of this may seem obvious to the point of being silly. **However, one of my co-authors once told me that many definitions that are so mundane as to seem silly are actually the core of ML Theory**. Although I don’t know ML Theory, I only have humility in my heart, so I will not omit these ordinary definitions.

Finally, this series of notes is in my Awesome-ML-Sys warehouse. I will also think about some issues from the SYS perspective and urge myself to learn.

## Introduction

- **The difference between RL and supervised learning**

To be honest, this is something I haven't figured out since my undergraduate degree. The teacher's courseware here is very good.

1. The input data is sequential, and the order has a major impact.
2. The learner will not be told what action to take, and needs to discover the action with the greatest ultimate benefit by itself.
3. Trial and error exploration requires a trade-off between exploration (exploring new strategies) and exploitation (utilizing current strategies)
4. There is no supervisor, only reward signal, and the reward signal is delayed.

> The environment that interacts with an agent oriented to a decision-making task is a dynamic random process. The distribution of its future state is jointly determined by the current state and the action of the agent's decision-making. Each round of state transition is accompanied by two aspects of randomness: one is the randomness of the action of the agent's decision-making, and the other is the randomness of the environment that samples the state of the next moment based on the current state and the action of the agent. By describing the dynamic random process of the environment, we can clearly feel that learning in a dynamic random process is very different from learning under a fixed data distribution.

- **Properties of Reinforcement Learning**

1. trial and error exploration
2. delayed reward
3. time matters, sequential decision making, not i.i.d.
4. Actions affect the environment

- **Possible issues with RL**

1.interpretability
2. diversity of the environment
3. overfitting on training environment
4. reward engineering
5. no safe guarantee
6. low sample efficiency

## RL Basic

### Common elements of RL

1. Agent and Environment
2. State
3. Observation: Note that state and observation will be distinguished in [next paragraph](#Seqential-Decision-Making)
4. Reward: reward is a scalar feedback given by the environment, which represents the performance of the agent at the current time step t; all reinforcement learning goals can be summarized as maximizing the cumulative expected reward

### Sequential Decision Making

The goal of the Agent is to select a series of actions to maximize the cumulative expected reward, and the Actions it selects need to have a long-term impact. Reward is often delayed, and the Agent needs to make a trade-off between immediate reward and long-term reward.

- History (history, $H_t$): History is a sequence consisting of observations (Observations), actions (Actions) and rewards (Rewards), denoted as
$H_t = O_1, A_1, R_1, O_2, A_2, R_2, \ldots, O_t, A_t, R_t$. It records the entire interaction process between the agent and the environment from time 1 to t. Although in a fully observable MDP, future states and rewards only depend on the current state and actions and do not require the complete $H_t$, the history can be used as a learning record or debugging tool for the agent.

- Environment State (Environment State, $S_{t}^{e}$): The true and complete state of the environment at a certain moment, describing all relevant information of the environment (such as all locations in the maze, walls, exits, etc.). In MDP, the agent can directly observe $S_{t}^{e}$, and the state transition only relies on the current $S_{t}^{e}$ and action $A_t$, not the history.

- Observation (observation, $O_t$): the information received by the agent from the environment. In a fully observable MDP, observation is equal to the environment state, that is, $O_t = S_{t}^{e}$, because the environment is fully observable and the agent can see the entire content of the environment.

- Agent State (agent state, $S_{t}^{a}$): the agent’s representation of the environment. In a fully observable MDP, the agent state is directly equal to the environment state, that is, $S_{t}^{a} = O_t = S_{t}^{e}$, because the agent can directly see the true state of the environment.

### Main components of RL Agent

1. Model

The model predicts what will happen next in the environment. It can predict the state of the environment in the next step, or the reward of the environment.

2. Policy

After the model predicts what will happen next, the policy is responsible for taking specific actions, which are divided into stochastic policy and deterministic policy.

3. Value Function

The value function is used to evaluate the long-term return of a state or a state-action pair under a specific strategy, and indirectly determine which strategy is better. There are two specific forms:

- State Value Function is used to evaluate how good a certain state $s$ is, representing the expected discounted reward of following policy $\pi$ in state $s$. The formula is:

$$
v_{\pi}(s) = E_{\pi}[G_t | S_t = s] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s]
$$

Among them, $G_t$ is the sum of future rewards starting from time $t$, $R_{t+k+1}$ is the reward at each time step, and $\gamma$ is the discount factor used to measure the importance of long-term rewards.

- Q-function (Q-function, state-action value function) is used to evaluate how good it is to take action $a$ in state $s$, representing the expected discounted reward of taking action $a$ in state $s$ following policy $\pi$, and is used to choose between multiple actions. The formula is:

$$
q_{\pi}(s, a) = E_{\pi}[G_t | S_t = s, A_t = a] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a]
$$The core role of a value function (whether $V(s)$ or $Q(s,a)$) is to evaluate the long-term payoff of a state or state-action pair under a particular strategy, rather than directly evaluating the strategy itself. It provides a basis for optimizing strategies. By comparing the value functions under different strategies, a strategy with better performance can be selected.

## Classification of RL

### Value-Based (value-based)

- **Definition**: Value-Based Agent mainly makes decisions by learning the "value function". The value function measures the goodness of a state (or state-action pair) and represents the expected cumulative reward that can be obtained if you start from that state and act according to a certain strategy. For example, $V(s)$ is the value of state $s$, and $Q(s, a)$ is the value of executing action $a$ in state $s$.
- **Features**: The agent does not directly learn "how to do it" (strategy), but learns "what is best" (valuation of value). Then implicit strategies are derived based on the value function (such as choosing the action with the highest value).
- **Learning Goal**: Learn an accurate value function (such as $Q(s, a)$) in order to know the long-term reward for each state or action pair. Finally, the implicit strategy is indirectly generated through the value function.
- **Example**: Q-Learning and Deep Q-Network (DQN) are both typical Value-Based methods. They learn a table or function of $Q$ values ​​and then choose an action based on the $Q$ values.

### Policy-Based (based on policy)

- **Definition**: Policy-Based Agent directly learns a policy (policy), that is, the mapping from state to action (can be a probability distribution). The policy can be explicit, such as "In state $s$, choose action $a_1$ with 80% probability and choose $a_2$ with 20% probability".
- **Features**: Agent does not rely on the value function, but directly optimizes the strategy to maximize the cumulative reward. Policies can be parameterized (such as neural networks), tuned via gradient descent, etc.
- **Learning Goal**: Directly learn the optimal strategy (such as $\pi(a|s)$, which represents the probability of selecting action $a$ in state $s$) to maximize long-term rewards. No value function is required (although it is sometimes used in combination).
- **Example**: The REINFORCE algorithm is a typical Policy-Based method that directly optimizes policy parameters.

### Actor-Critic (actor-critic)
- **Features**: Actor-Critic combines the advantages of Value-Based and Policy-Based. It has two parts:
  - **Actor**: Responsible for learning strategies (similar to Policy-Based) and directly outputting actions or action probabilities.
  - **Critic**: Responsible for learning the value function (similar to Value-Based), evaluating the quality of the current strategy (such as $Q(s, a)$ or $V(s)$), and guiding Actor improvement.
- **Learning Goal**: Learn the policy (Actor) and the value function (Critic) simultaneously. The goal is to optimize the policy to maximize long-term rewards, while using the value function to provide more accurate feedback.
- **Example**: A3C (Asynchronous Advantage Actor-Critic) or PPO (Proximal Policy Optimization) are common Actor-Critic methods.


### Model-Based (based on model)

- **Definition**: Model-Based Agent makes decisions by learning a model of the environment. The model includes state transition rules (such as "if action $a$ is performed in state $s$, it will be transferred to state $s'$") and reward functions ("how much reward can be obtained by performing this action"). In layman's terms, this is like making a travel guide before traveling and basically deciding on the route.
- **Feature**: The agent learns not only policies or values, but also the internal structure (model) of the environment. Once you have a model, you can choose the best action through planning or simulation instead of relying entirely on trial and error.
- **Learning Goal**: Learn an accurate model of the environment (state transitions and rewards), and then use the model to optimize a policy or value function to maximize long-term rewards.
- **Example**: When using Dynamic Programming or AlphaGo's Monte Carlo Tree Search (MCTS), the model is the key to predicting future states and guiding action selection.

### Model-Free (model-free)

- **Definition**: Model-Free Agent does not learn the environment model, but directly learns the strategy or value function from experience (trial and error). They only care about the currently observed states, actions, and rewards, and do not need to predict future state transitions or rewards. It's like traveling randomly in a city without any advance strategy.
- **Features**: Simple and direct, low computational cost, but may require more samples (experience) to learn because of the lack of guidance from the environment structure.
- **Learning Objective**: Directly learn the policy (Policy-Based) or value function (Value-Based) to maximize the cumulative reward without relying on the environment model.
- **Example**: Q-Learning (Value-Based), REINFORCE (Policy-Based), PPO or DQN are all Model-Free methods.

| Type | What to learn | Model | Value function | Strategy | Advantages | Disadvantages |
|----------------|--------------------------------|-------------------|--------------------------|-------------------|--------------------------------|--------------------------------|
| Value-Based | Value function (indirectly derived strategy) | Not required | Yes | Implicit | Simple, high sample efficiency | Strategy may be inflexible |
| Policy-Based | Strategy | Not required | No | Yes | The policy is flexible and can handle continuous actions | The sample efficiency is low and the training is unstable |
| Actor-Critic | Strategy + Value Function | Not required | Yes | Yes | Combining the advantages of both, stable and efficient | Complex, difficult to adjust parameters |
| Model-Based | Environment model + strategy/value | Required | Optional | Optional | High sample efficiency, capable of planning | Difficulty in model learning, high computational cost |
| Model-Free | Strategy or value function | Not required | Optional | Optional | Simple, easy to implement | Low sample efficiency, relying on a lot of trial and error |## Multi-armed bandit problem

Reinforcement learning focuses on learning during the interaction between the agent and the environment, which is a trial-and-error learning paradigm. The multi-armed bandit problem is a simplified version of reinforcement learning. Multi-armed bandits do not have state information, only actions and rewards. It is the simplest form of "learning in interaction with the environment".

For the multi-armed bandit problem, the winning probability of each lever is certain, but it is unknown to the agent. What the agent needs to do is to select a rod at each time step, and then receive rewards based on the rod's winning probability, ultimately maximizing the cumulative reward. One of the simplest strategies is to always take the first action, but this relies heavily on luck. If you have great luck, you may pull the lever that can get the maximum expected reward, that is, the optimal lever; but if you have bad luck, you may get the minimum expected reward. It can be seen that there is a serious conflict in the balance between exploration and utilization. Exploration refers to trying to pull more possible levers. This lever may not necessarily get the biggest reward, but this method can find out the awards of all levers. For example, we need to pull all the levers once (even many times) to know which lever may get the biggest reward. Exploitation refers to pulling the lever with the largest known expected reward. Since the known information only comes from a limited number of interactive observations, the current optimal lever is not necessarily the global optimal. For example, for a 10-arm bandit, we have only pulled 3 of the levers, and then we keep pulling the lever with the largest expected reward among the 3 levers, but it is very likely that the lever with the largest expected reward is among the remaining 7. Even if we try each of the 10 levers 20 times, we find that the empirical expected reward of lever 5 is the highest, but there is still a slight probability that the other 6 The true expected reward for lever number 5 is higher than lever number 5.

In order to have a deeper understanding of the bandit problem, the concept of regret is introduced here: the difference between the expected reward obtained by pulling the current lever $\alpha$ and the expected reward of the optimal lever, that is, $R(\alpha) = Q^* - Q{(\alpha)}$. Cumulative regret is the total amount of regret accumulated after operating the lever $t$ times. For a complete step decision, the cumulative regret is $\sigma_R = \sum_{i=1}^{t} (Q^* - Q{(\alpha_t)})$. Maximizing cumulative reward is equivalent to minimizing cumulative regret.

### $\epsilon$-greedy algorithm

Q(a) is the value function of action (action a), which represents the average reward that the agent expects to receive if it chooses a certain action a. Specifically, $Q_t(a)$ is the sum of rewards obtained each time action $a$ is chosen until step $t$, divided by the number of times $a$ is chosen. This is an empirically based average used to estimate how "good" an action $a$ is.

$$
  Q_t(a) = \frac{\text{sum of rewards when action a was taken prior to t}}{\text{number of times a was taken prior to t}}
$$

For example, action A sometimes gives you 10 points and sometimes it gives you 2 points. You attempted Action A 5 times, earning a total of 40 points. Then \( Q(A) = 40 / 5 = 8 \) points. This is the current estimated value of action A.

With the definition of $Q_t(a)$, we can define a Greedy Strategy, that is, choosing the action $a$ that maximizes $Q_t(a)$ each time. This strategy is actually the local minima, falling into the Exploration-Exploitation Dilemma. To this end, the $\epsilon$-greedy strategy introduces exploration. It uses a greedy strategy to choose the best action most of the time, but with a small probability $\epsilon$ randomly chooses an action (selected uniformly from all possible actions).

A classic pseudocode can be given here. Consider a slot machine with k levers (arms or actions), and the agent needs to decide which lever to pull each time to maximize the reward. The ε-greedy strategy is used here to balance exploration and exploitation.

```plaintext
1: for a = 1 to k do
2: Q(a) = 0, N(a) = 0
3: end for
4: loop
5: A = {
         arg max_a Q(a) with probability 1 - ε
         uniform(A) with probability ε
       }
6: r = bandit(A)
7: N(A) = N(A) + 1
8: Q(A) = Q(A) + 1/N(A) [r - Q(A)]
9: end loop
```

1 to 3 are initialized and the value of each action is learned from scratch. 4 to 9 are the main loop:

- Action selection: Make an $\epsilon$-greedy selection as described above.
- Get the reward: After selecting action A, pull down the lever to get the reward r.
- Update the number of selections: Each time action A is selected, increase its number of selections $N(A)$ by 1. to track how often each action is attempted.
- Update action value: Use an incremental update formula to adjust $Q(A)$ based on the newly earned reward $r$ and the current estimate $Q(A)$. In the formula, $1/N(A)$ is the "learning rate" or "step size". As $N(A)$ increases (that is, the action is selected more times), the step size becomes smaller and the update becomes smoother. $r - Q(A)$ is the "error" or "bias": the new reward $r$ minus the current estimate $Q(A)$. If $r > Q(A)$, the action is better than we estimated; if $r < Q(A)$, the estimate is too high.
- The loop continues, repeatedly selecting actions, obtaining rewards, and updating estimates.

More specifically:

$$
  Q_t(a_t) = Q_{t-1} + \frac{1}{N_t(a_t)} (r_t - Q_{t-1}(a_t))\\
  \text{NewEstimate} = \text{OldEstimate} + \text{StepSize} \times (\text{Target} - \text{OldEstimate})
$$

This is a more formal expression of the formula in line 8, clarifying the time step dependence. $Q_t(a_t)$ is the estimate of action $a_t$ at time step $t$. $Q_{t-1}(a_t)$ is the estimate (old estimate) of the same action at time step $t-1$. $r_t$ is the reward obtained after selecting $a_t$ at time step $t$. $N_t(a_t)$ is the total number of times action $a_t$ has been selected up to time step $t$. The formula is the same as line 8, indicating that the new estimate is obtained by adding an adjustment to the old estimate based on the error $ [r_t - Q_{t-1}(a_t)] $ and the learning rate $ \frac{1}{N_t(a_t)} $ .

Of course, this multi-armed bandit model is the most simplified RL task because rewards are not delayed, state does not change, and there is no subsequent impact between actions.

Consider briefly the cumulative regret of an $\epsilon$-greedy strategy in a multi-armed bandit. After a long enough time, the $\epsilon$-greedy strategy can completely explore the expectations of all levers. As long as it does not fall within the probability of $\epsilon$, it will definitely choose the globally optimal lever without regret. If it falls within the probability of $\epsilon$, the lever will be pulled randomly, and the accumulated regret will be approximately a straight line with a slope proportional to $\epsilon$.Therefore, we can expect a picture like this:

![](./pics/epsilon-regrets.png)

If we set $\epsilon$ to $\frac{1}{t}$, then the performance of accumulated regret is, as you would expect, better than the fixed-value $\epsilon$-greedy strategy.