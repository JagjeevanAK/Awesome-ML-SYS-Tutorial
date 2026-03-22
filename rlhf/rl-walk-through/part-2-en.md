# Markov decision process

The book continues from the previous article. Unexpectedly, there was a whole week between the two writings of this note. What is it that keeps interrupting my study progress?

According to a friend, as a project leader, I care too much about the right to speak. The right to speak is the greatest right, but it also prevents me from sharing the things at hand with others. Based on this, I decided to spend at least 1/3 of my time every day, shut down WeChat and slack, concentrate on studying on my own, and ponder some things.

## Markov process

Markov process means "at this moment, the past will not affect the future", this abstraction can model quite a lot of practical problems. Unlike the multi-armed bandit problem, the Markov decision process contains state information and a transition mechanism between states. If you want to use reinforcement learning to solve a practical problem, the first step is to abstract the practical problem into a Markov decision process.

> Being Markovian does not mean that this random process has nothing to do with history. Because although the state at time $t$ is only related to the state at time $t-1$, the state at time $t-1$ actually contains the information about the state at time $t-2$. Through this chain relationship, historical information is passed to the present. Markov properties can greatly simplify operations, because as long as the current state is known, all historical information is no longer needed, and the future can be determined using the current state information.

Markov process refers to a random process with Markov properties, also known as Markov chain. We usually use the set $(S, P)$ to describe a Markov process, where $S$ is a finite set of states and $P$ is the state transition matrix. Suppose there are $n$ states in total, at this time $S = \{s_1, s_2, \cdots, s_n\}$. The state transition matrix $P$ marks the transition probability between states, that is:

$$
P = \begin{bmatrix}
P(s_1|s_1) & \cdots & P(s_n|s_1) \\
\vdots & \ddots & \vdots \\
P(s_1|s_n) & \cdots & P(s_n|s_n)
\end{bmatrix}
$$

Element $P(s_j|s_i) = P(S_{t+1} = s_j|S_t = s_i)$ in row $i$ and column $j$ in matrix $P$ represents the probability of transitioning from state $s_i$ to state $s_j$. We call $P(s|s)$ the state transition function. Starting from a certain state $s_i$, the sum of the probabilities of reaching other states should be 1, that is, the sum of each row of the state transition matrix is ​​1.

From a Markov process, we can start from a certain state and generate a state sequence (episode) according to the transition matrix state transition matrix. This sequence is also called a sample path (sampling). For example, the sequence $s_1 \to s_2 \to s_3 \to s_6$ or the sequence $s_1 \to s_1 \to s_2 \to s_3 \to s_4 \to s_5 \to s_3 \to s_6$ may occur. The probability of generating these sequences is related to the state transition matrix.### Reward function

By adding the reward function $r$ and the discount factor $\gamma$ to the Markov process, you can get the Markov reward process. A Markov reward process consists of $(S, P, r, \gamma)$, where the meaning of the component elements is as follows.

- $S$ is a finite set of states.
- $P$ is the state transition matrix.
- $r$ is the reward function, and the reward $r(s)$ of a certain state $s$ refers to the expectation of the reward that can be obtained when transitioning to that state.
- $\gamma$ is the discount factor, ranging from (0, 1).

The reason for introducing the discount factor is that forward benefits have certain uncertainty. Sometimes we prefer to get some rewards as soon as possible, so we need to discount the forward benefits. A $\gamma$ close to 1 is more concerned with long-term cumulative rewards, and a $\gamma$ close to 0 is more interested in short-term rewards.

In a Markov reward process, starting from the state $S_t$ at the t-th time until the termination state, the sum of the attenuation of all rewards is called the return $G_t$ (Return), and the formula is as follows:

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$

### Value function

In the Markov reward process, the expected return of a state (that is, the expectation of future cumulative rewards starting from this state) is called the value of this state. The values ​​of all states constitute a value function. The input of the value function is a certain state, and the output is the value of this state. We express the value function as $V(s) = \mathbb{E}[G_t|S_t = s]$. According to the definition of return, we can get:

$$
V(s) = \mathbb{E}[G_t|S_t = s] \\
= \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots | S_t = s] \\
= \mathbb{E}[R_t + \gamma (R_{t+1} + \gamma R_{t+2} + \cdots) | S_t = s] \\
= \mathbb{E}[R_t + \gamma G_{t+1} | S_t = s] \\
= \mathbb{E}[R_t + \gamma V(S_{t+1}) | S_t = s]
$$

On the one hand, the expectation of immediate reward is exactly the output of the reward function, that is, $\mathbb{E}[R_t|S_t = s] = r(s)$; on the other hand, the shelling part of learning $\mathbb{E}[\gamma V(S_{t+1})|S_t = s]$ can be calculated based on the transition probability starting from state $s$, that is, we can get$$
V(s) = r(s) + \gamma \sum_{s' \in S} p(s'|s)V(s')
$$

The above is the very famous Bellman equation in the Markov reward process, which holds true for every state. Consider a Markov reward process with a total of $n$ states, that is, $S = \{s_1, s_2, \cdots, s_n\}$. We represent the values of all states as a column vector $V = [V(s_1), V(s_2), \cdots, V(s_n)]^T$. In the same way, the reward function is turned into a column vector $R = [r(s_1), r(s_2), \cdots, r(s_n)]^T$. So we can write the Bellman equation in matrix form:

$$
V = R + \gamma P V
$$

$$
\begin{aligned}
\begin{bmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_n)
\end{bmatrix} &=
\begin{bmatrix}
r(s_1) \\
r(s_2) \\
\vdots \\
r(s_n)
\end{bmatrix} + \gamma
\begin{bmatrix}
P(s_1|s_1) & P(s_2|s_1) & \cdots & P(s_n|s_1) \\
P(s_1|s_2) & P(s_2|s_2) & \cdots & P(s_n|s_2) \\
\vdots & \vdots & \ddots & \vdots \\
P(s_1|s_n) & P(s_2|s_n) & \cdots & P(s_n|s_n)
\end{bmatrix}
\begin{bmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_n)
\end{bmatrix}
\end{aligned}
$$

We can directly write the matrix operation as a transfer requirement and get the following analytical solution:

$$
V = R + \gamma P V
$$

$$
(I - \gamma P)V = R
$$

$$
V = (I - \gamma P)^{-1} R
$$

The computational complexity of the above analytical solution is $O(n^3)$, and the main cost comes from matrix inversion. Therefore this method is only suitable for Markov reward processes with small states. When solving the value function in a larger-scale Markov reward process, dynamic programming algorithms, Monte-Carlo methods, and temporal differences can be used.## Markov decision process

Markov processes are spontaneous stochastic processes, and the actions of an agent may affect the environment. When we add actions to MRP, we get MDP. A Markov decision process consists of $(S, A, P, r, \gamma)$, where the meaning of the component elements is as follows.

- $S$ is a collection of states;
- $A$ is a collection of actions;
- $\gamma$ is the discount factor;
- $r(s, a)$ is the reward function. At this time, the reward can depend on both the state $s$ and the action $a$. When the reward function only depends on the state $s$, it degenerates to $r(s)$;
- $P(s'|s, a)$ is the state transition probability, which represents the probability of reaching state $s'$ after performing action $a$ in state $s$.


Note that in the definition of MDP above, we no longer use the state transition matrix method similar to the definition of MRP, but directly express it as a state transition function. This is done first because the state transition is also related to the action at this time and becomes a three-dimensional array instead of a matrix (two-dimensional array); secondly, because the state transition function has more general meaning. For example, if the state set is not limited, it cannot be represented by an array, but it can still be represented by a state transition function.

### Strategy

The agent's policy for selecting actions is usually represented by the letters $\pi$. Policy $\pi(a|s) = P(A_t = a|S_t = s)$ is a function representing the probability of choosing action $a$ in state $s$. Strategies can be divided into two categories:

- **Deterministic policy**: When a policy is deterministic, it only chooses a certain action in each state.
- **Stochastic policy**: In stochastic policy, $\pi(a|s)$ represents the probability of choosing action $a$ from state $s$. Stochastic strategies allow the selection of multiple possible actions in a given state, each action having a certain probability distribution.

In a Markov decision process (MDP), due to the existence of Markov properties, the policy only needs to decide the action $a$ based on the current state $s$, without relying on past actions or state sequences. Therefore, the policy in MDP can be viewed as the mapping function $\pi: S \to A$ from state to action. In MDP, the transition probability of the current state is jointly determined by the policy $\pi$ and the state transition probability $P(s'|s, a)$.

### Value function

In reinforcement learning, the goal of policy $\pi$ is to maximize the long-term cumulative reward (discounted cumulative reward). Therefore, the optimization problem of the policy can be expressed as finding the optimal policy $\pi^*$ to maximize the expected return $V^\pi(s)$ or $Q^\pi(s, a)$.Different from MRP, in MDP, due to the existence of actions, we additionally define an action-value function. We use $Q^{\pi}(s, a)$ to represent that in MDP following the policy $\pi$, executing action $a$ for the current state $s$ will get the expected return:

$$
Q^{\pi}(s, a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]
$$

The relationship between the state value function and the action value function: In using policy $\pi$, the value of state $s$ is equal to the sum of the probabilities of taking all actions in that state based on policy $\pi$ and the value of the corresponding action:

$$
V^{\pi}(s) = \sum_{a \in A} \pi(a|s) Q^{\pi}(s, a)
$$

Using policy $\pi$, the value of taking action $a$ in state $s$ is equal to the immediate reward plus all possible next state states after decay.

The product of state transition probabilities and corresponding values:

$$
Q^{\pi}(s, a) = r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^{\pi}(s')
$$

### Bellman expectation equation

Based on this, we continue to derive the Bellman Expectation Equation. It looks very complicated, but it is actually based on the basic definitions:

$$
V^{\pi}(s) = E_{\pi}[R_t + \gamma V^{\pi}(S_{t+1})|S_t = s]
$$

$$
V^{\pi}(s) = \sum_{a \in A} \pi(a|s) \left( r(s,a) + \gamma \sum_{s' \in S} p(s'|s,a)V^{\pi}(s') \right)
$$

$$
Q^{\pi}(s,a) = E_{\pi}[R_t + \gamma Q^{\pi}(S_{t+1}, A_{t+1})|S_t = s, A_t = a]
$$

$$
Q^{\pi}(s,a) = r(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) \sum_{a' \in A} \pi(a'|s') Q^{\pi}(s', a')
$$