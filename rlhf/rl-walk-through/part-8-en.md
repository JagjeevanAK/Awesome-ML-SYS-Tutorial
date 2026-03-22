# imitation learning

Although reinforcement learning does not require labeled data in supervised learning, it relies heavily on the setting of the reward function. Sometimes making some small changes in the reward function can make a big difference in the trained strategy. In many real-life scenarios, the reward function is not clear at a glance, and the design of the reward function requires a lot of trial and error and debugging processes. Fortunately, we can use the strategies of human experts to guide the agent's strategy training, thus accelerating the training process. Based on this idea, we can extract strategic knowledge from the trajectories of human experts for the agent to learn.

We can think of this process as a problem of imitation learning (imitation learning) research model. Under the framework of imitation learning, the expert can provide a series of state action pairs $\{(s_t, a_t)\}$, which represents the action $a_t$ taken by the expert in the state $s_t$. The goal of imitation learning is to train a strategy $\pi_{\theta}$ as close as possible to the strategy of human experts, so as to better complete the task.

1. Behavior cloning (BC)
2. Inverse RL
3. Generative adversarial imitation learning (GAIL)

## Behavior cloning

Behavior cloning (BC) directly uses the supervised learning method to input the action samples of $(s_t, a_t)$ in the expert data, a is regarded as the label, and the learning goal is:

$$
\theta^* = \arg \min_{\theta} \mathbb{E}_{(s,a) \sim B} [\mathcal{L}(\pi_{\theta}(s), a)]
$$

Among them, B is the expert's data set, and C is the loss function under the corresponding supervised learning framework. If the action is discrete, the loss function can be estimated by maximum likelihood. If the action is continuous, the loss function can be the mean square error.

During the training process, BC can quickly learn a good strategy. For example, AlphaGo first learned how human players play chess from 30 million step sub-data of 160,000 chess games. Using this behavioral cloning method alone, AlphaGo's chess power has surpassed that of many amateur Go players. Since the implementation of BC is very simple, it can be used as a policy pre-training method in many practical scenarios. BC can enable the strategy to obtain good initial performance in a short period of time, and by imitating expert data, it can quickly converge to a better strategy without excessive exploration. However, there are some problems with behavioral cloning, one of which is the compounding error problem, which will not be discussed here.

## Generative adversarial imitation learning

Generative adversarial imitation learning inherits the essential logic and ideas of generative adversarial networks. GAIL is actually a form of imitation learning. Different from imitation learning based on behavioral cloning, there is a discriminator and a strategy in the GAIL algorithm. The strategy needs to know that the occupancy metric $\rho_{\pi}(s, a)$ of the environmental action on $(s, a)$ is consistent with the occupancy metric of the expert strategy. To achieve this goal, the strategy needs to interact with the environment to collect information about the next state and take further actions. This is different from BC, which does not need to interact with the environment at all. There are two discriminators and a strategy in the GAIL algorithm. The strategy is regarded as the generator in the generative adversarial network. Given a state, the strategy will output the action that should be taken in a state, and the discriminator (discriminator) $D$ takes the state-action pair $(s, a)$ as input and outputs a value between 0 and 1, indicating that the discriminator thinks the action-state pair $(s, a)$ Whether it is data from an agent or an expert, the purpose of the discriminator is to train an optimal discriminator, and the training goal is to perform gradient descent on the internal strategy of the agent. Therefore, the loss function using the discriminator $D$ is:$$
\mathcal{L}(\phi) = -\mathbb{E}_{\rho_{\pi}} [\log D_{\phi}(s, a)] - \mathbb{E}_{\rho_E} [\log(1 - D_{\phi}(s, a))]
$$

where $\phi$ are the parameters of the discriminator. After the discriminator $D$, if the occupancy frequency of the imitator's strategy is such that the trajectories produced by its interaction can be mistaken by the discriminator as trajectories produced by expert trajectories. So there is no need to worry about the deviation of the occupancy frequency. We can use the output of the discriminator $D$ as the reward function to train the policy. Specifically, if the state action pair $(s, a)$ of the environment is input to the discriminator $D$, the value of $D(s, a)$ will be output, and then the reward is set to:

$$
r(s, a) = -\log D(s, a)
$$

Therefore, we can use any reinforcement learning algorithm and use these data training strategies. The data distribution of the imitator strategy will be close to the real expert distribution. In the end, the generator cannot be well distinguished by the discriminator in the real environment, achieving the self-standard of imitation learning.