# Dyna-Q and DQN algorithms

In reinforcement learning, "model" usually refers to the environment model that interacts with the agent, that is, modeling the state transition probability and reward function of the environment. According to whether there is an environment model, reinforcement learning algorithms are divided into two types: model-based reinforcement learning (model-based reinforcement learning) and model-free reinforcement learning (model-free reinforcement learning). Model-free reinforcement learning directly performs strategy improvement or value estimation based on the data sampled from the interaction between the agent and the environment. Temporal difference algorithms (Sarsa and Q-learning algorithms) are two model-free reinforcement learning methods. In model-based reinforcement learning, the model can be known in advance, or it can be learned based on data sampled from the interaction between the agent and the environment, and then this model can be used to help strategy improvement or value estimation. Two representative dynamic programming algorithms, namely policy iteration and value iteration, are model-based reinforcement learning methods, in which the environment model is known in advance. This summary introduces the Dyna-Q algorithm, which is also a very basic model-based reinforcement learning algorithm, but its environment model is estimated by sampling data.

The reinforcement learning algorithm has two important evaluation indicators: one is the expected return of the algorithm's converged strategy in the initial state, and the other is the sample complexity, which is the number of samples that the algorithm needs to sample in the real environment to achieve convergence results. Since the model-based reinforcement learning algorithm has an environment model, the agent can additionally interact with the environment model, and the demand for samples in the real environment is often reduced, so it usually has lower sample complexity than the model-free reinforcement learning algorithm. However, the environment model may not be accurate and cannot completely replace the real environment. Therefore, the expected return of the strategy after the model-based reinforcement learning algorithm converges may not be as good as the model-free reinforcement learning algorithm.

## Dyna-Q

The Dyna-Q algorithm is a very classic model-based algorithm. It uses a method called Q-planning to generate some simulated data based on the model, and then uses the simulated data and real data to improve the strategy. Q-planning selects a previously visited state each time, takes an action that has been performed in that state, obtains the transferred state and rewards through the model, and uses the Q-learning update method to update the action value function based on this simulation data. To put it simply, the Dyna-Q algorithm will perform one step of real sampling and multiple steps of simulated sampling with the model in each iteration, and then use simulated sampling data and real sampling data to improve the strategy together.



- Initialize $Q(s, a)$, initialize model $M(s, a)$
- for sequence e = 1 -> $E$ do:
  - Get the initial state s
  - for t = 1 -> $T$ do:
    - Use $\epsilon$-greedy strategy to select action a in current state s according to Q
    - Get environment feedback r, s'
    - $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
    - $M(s, a) \leftarrow r, s'$
    - for times n = 1 -> $N$ do:
      1. Randomly select a previously visited state $s_m$
      2. Get an action $a_m$ that has been executed in state $s_m$
      3. $r_m, s'_m \leftarrow M(s_m, a_m)$
      4. $Q(s_m, a_m) \leftarrow Q(s_m, a_m) + \alpha [r_m + \gamma \max_{a'} Q(s'_m, a') - Q(s_m, a_m)]$
    - end for
    - $s \leftarrow s'$
  - end for- end for

It can be seen that after each environmental feedback execution and interactive execution of Q-learning, Dyna-Q will perform n times of Q-planning, where the number of Q-planning N is an adjustable parameter. When it is currently 0, it is ordinary Q-learning. It is worth noting that the above Dyna-Q algorithm is executed in a simple and deterministic environment, so when seeing a piece of empirical data $(s, a, r, s')$, the model can be directly projected to make updates, that is, $M(s, a) \leftarrow r, s'$.

## DQN

The Q learning algorithm needs to store a table of all action $Q$ values in each state. When the state space is too large, this table will be very large. In fact, in many cases, actions and states are not discrete, and it is impossible to exhaustively enumerate them. Therefore, we can only estimate the $Q$ value by fitting the function. Here we introduce DQN, which is used to solve the problem of **discrete** actions in **continuous** states.

CartPole is a very classic continuous state and discrete action problem. The task of the agent is to keep the pole on the car vertical by moving left and right. If the inclination of the pole is too large, or the car deviates too much from the initial position, or the persistence time reaches 200 frames, the game will end. The state of the agent is a vector with 4 dimensions. Each dimension is continuous. Its actions are discrete and can only move left or right. The action space size is 2.

Assume that the action value function of the car is $Q(s, a)$. Since the state is continuous and cannot be recorded in a table, a common solution is to use the idea of ​​function approximation, so we can use a neural network to represent $Q$. If the action is continuous (infinite), the input of the neural network is the state $s$ and the action $a$, and then outputs a scalar representing the value that can be obtained by taking the action $a$ in the state $s$. If the action is discrete (limited), in addition to the approach in the case of action connection, we can also make it output the $Q$ value of each action at the same time after the state $s$ is input to the neural network. Popular $DQN$ (and $Q$-learning) can only handle the case of discrete actions, because there is $\max_a$ in the update process of the graph number $Q$. Assume that the reference used by the neural network to fit the graph number $w$ is that we can express the $Q$ values ​​of all possible actions $a$ under a single state $s$ as $Q_w(s, a)$. We call the neural network used to fit the graph number a $Q$ network. Now let’s review the update rules of Q-learning to construct the loss function of the Q-network:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a' \in A} Q(s', a') - Q(s, a) \right]$$The above formula uses the temporal difference (TD) learning objective $r + \gamma \max_{a' \in A} Q(s', a')$ to incrementally update $Q(s, a)$, that is, to make $Q(s, a)$ and the TD objective $r + \gamma \max_{a' \in A} Q(s', a')$ close. Therefore, for a set of data [$(s_i, a_i, r_i, s'_i)$], we can naturally construct the loss function of the Q network in the form of mean square error:

$$\omega^* = \arg \min_{\omega} \frac{1}{2N} \sum_{i=1}^N \left[ Q_{\omega}(s_i, a_i) - \left( r_i + \gamma \max_{a'} Q_{\omega}(s'_i, a') \right) \right]^2$$

At this point, we can extend Q-learning to a neural network form—the deep Q network (DQN) algorithm. Since DQN is an off-policy algorithm, we can use an $\epsilon$-greedy strategy to balance exploration and utilization when collecting data, store the collected data, and use it in subsequent training. There are two very important modules in DQN - experience replay and target network, which can help DQN achieve stable and excellent performance.

### Experience replay

In general supervised learning, assuming that the training data is independently and identically distributed, each time we train a neural network, we randomly sample one or several data from the training data to perform gradient descent. As learning continues, each training data will be used multiple times. In the original Q-learning algorithm, each data is only used to update the value once. In order to better combine Q-learning with deep neural networks, the DQN algorithm adopts the experience replay method. The specific method is to maintain a replay buffer and store the four-tuple data (state, action, reward, next state) sampled from the environment each time into the replay buffer. When training the Q network, a number of data are randomly sampled from the replay buffer for training. Doing so can serve the following two purposes.

(1) Make the sample satisfy the independence assumption. In MDP, the data obtained by interactive sampling does not satisfy the independence assumption, because the state at this moment is related to the state at the previous moment. Non-independent and identically distributed data has a great impact on training neural networks, making the neural network fit to the most recently trained data. Using experience replay can break the correlation between samples and make them satisfy the independence assumption.

(2) Improve sample efficiency. Each sample can be used multiple times, which is very suitable for gradient learning of deep neural networks.> Note that Q-learning is an off-policy algorithm, but the off-policy algorithm does not mean that experience will be used repeatedly.

### Target network

The final update goal of the DQN algorithm is to make $Q_w(s, a)$ approach $r + \gamma \max_{a'} Q_w(s', a')$. Since the TD error target itself contains the output of the neural network, the target is constantly being changed while updating the network parameters, which is very easy to cause instability and oscillation in neural network training. In order to solve this problem, DQN uses the idea of ​​​​the target network: Since the continuous updating of the $Q$ network during the training process will cause the target to continue to change, it is better to temporarily fix the $Q$ network in the TD target. In order to implement this idea, we need to utilize two sets of $Q$ networks.

1. Train the network $Q_w(s, a)$, which is used to calculate $Q_w(s, a)$ in the original loss function $\frac{1}{2} [Q_w(s, a) - (r + \gamma \max_{a'} Q_w(s', a'))]^2$.

2. The target network $Q_{\hat{w}}(s, a)$ is used to calculate the value of the TD error target $r + \gamma \max_{a'} Q_{\hat{w}}(s', a')$, where $\hat{w}$ represents the parameters of the target network.

If the parameters of the two sets of networks are consistent, oscillation will still occur, so the parameters of the target network $\hat{w}$ can be updated regularly, for example, every few steps, that is, $\hat{w} \leftarrow w$. The purpose of this is to make the parameters of the target network relatively stable, thereby stabilizing the calculation of the TD error target.


- Initialize network $Q_w(s, a)$ with random initial parameters $w$
- Copy the same parameters $w^- \leftarrow w$ to initialize the target network $Q_{w^-}$
- Initialize experience replay pool $R$
- for sequence $e = 1 \to E$ do:
  - Get the initial state of the environment $s_1$
  - for time step $t = 1 \to T$ do:
    1. Select action $a_t$ based on the current network $Q_w(s, a)$ with $\epsilon$-greedy strategy
    2. Execute action $a_t$, obtain environment feedback $r_t$, and the environment state changes to $s_{t+1}$
    3. Store $(s_t, a_t, r_t, s_{t+1})$ in the experience replay pool $R$
    4. If there is enough data in $R$, sample $N$ data from $R$ $\{(s_i, a_i, r_i, s_{i+1})\}_{i=1,\dots,N}$
    5. for each data $i$ do:
      - Calculate the target value $y_i = r_i + \gamma \max_{a'} Q_{w^-}(s_{i+1}, a')$
      - Minimize the target loss $L = \frac{1}{N} \sum_i (y_i - Q_w(s_i, a_i))^2$, thereby updating the current network $Q_w$
    6. Update target network
  - end for
- end for