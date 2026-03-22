# Kimi K1.5: Successful Practice of Long Context RL

Thanks to the kimi team for this great article. Around the same time that DeepSeek open sourced R1, many friends recommended the K1.5 technical report to me. Due to the tedious work, I never had time to read. I happened to be on a business trip to Seattle recently, and on the way to and from LAX to SEA, I finally had time to read this work religiously. The more I read, the more I feel like scholars appreciate each other and it is rare to meet a close friend.

Since graduating from undergraduate degree, I have indeed felt that I have made great progress in teamwork and personal abilities. However, after enrolling as a Ph.D. student, I have not published any high-intensity work, which makes me feel anxious. After reading such a powerful article today, I felt very refreshed. I hope that I can participate in more important work with an open source spirit in the rest of my scientific research career. Isn't it better than xxxxx to have his name appear on the list of authors of this work? Of course, this brings up the problem of how to prove one's credit in large projects. Even so, I always believe that my ideas are still very helpful.

Having said so much, this article mainly reviews my thoughts on reading the K1.5 technical report. Since it is a technical report, this solid article covers all aspects from data, training methods to training systems. It is really lingering after reading it.

## RL Recipe

The training of K1.5 can be subdivided into four stages: pretrain, vanila SFT, long-CoT SFT and RL. The technical report mainly tells the story of the RL stage.

### RL prompt selection

High-quality RL prompts need to be diverse, balanced and accurate to evaluate. In order to determine the difficulty of each prompt, the author uses an SFT model to generate answers 10 times at a higher temperature, and uses the pass rate within 10 times as the difficulty to balance the samples during training. In addition, some complex reasoning questions can also guess the correct answer through wrong derivation. In order to avoid such reward hacking, the author further ensures that the reasoning path and final answer of each prompt can be accurately verified. The author first eliminated questions that are prone to such errors, such as multiple-choice questions, true-false questions, and proof questions. Then, the author further filters out some questions whose answers are easy to guess. Specifically, the model was given 8 chances, and if it could give a direct answer more than 1 time without a CoT, it was removed.

### Long-CoT SFT

The author built a multi-modal long-CoT warmup dataset through prompt engineering to allow the model to initially learn these reasoning capabilities, evaluation, reflection, exploration and planning.### Length Penalty

During long context RL training, if the output length of the model is not controlled, it is easy to observe a significant increase in answer length. While this leads to better performance, overly long inference processes are costly to train and infer, and humans are generally not prone to overthinking. Therefore, the author introduces a length penalty term to control the output length of the model:

$$ \text{len\_reward}(i) = \begin{cases} \lambda & \text{if } r(x, y_i, y^*) = 1 \\ \min(0, \lambda) & \text{if } r(x, y_i, y^*) = 0 \end{cases} $$

$$ \lambda = 0.5 - \frac{\text{len}(i) - \text{min\_len}}{\text{max\_len} - \text{min\_len}} $$

In the formula, $r(x, y_i, y^*)$ is the reward of the $i$th reasoning process, which can be simply understood as correctness.

Think about these situations separately:

1. The longest reasoning process, the correct answer. $len\_reward = -0.5$, inhibits the model from generating this inference process.
2. The longest reasoning process, wrong answer. $len\_reward = -0.5$, inhibits the model from generating this inference process.
3. The shortest reasoning process, the correct answer. $len\_reward = 0.5$, encourages the model to generate this inference process.
4. The shortest reasoning process, wrong answer. $len\_reward = 0$, has no impact on the model.

For the reasoning process with a length below $\frac{\text{max\_len} + \text{min\_len}}{2}$, if the answer is wrong, the length reward is 0. If the answer is correct, the length reward is greater than 0 and decreases with length. For reasoning paths that exceed this length, regardless of whether the answer is correct or incorrect, the same negative length reward will be given.

### Sampling strategy

Although RL itself has better sampling properties (more difficult problems provide larger gradients), its training efficiency is still limited. Some explicit prior sampling methods may lead to greater performance gains. The solutions adopted by the author include:

1. Course learning: Start training with simpler tasks and gradually transition to more challenging tasks. Due to the limited performance of the initial RL model, spending limited computing resources on very difficult problems often only produces few correct samples, resulting in low training efficiency. At the same time, the data used for training naturally contains difficulty labels, further making the increasing difficulty learning strategy intuitive and effective.
2. Prioritize sampling: Problems with lower success rates are sampled more times. We track the success rate $s_i$ for each problem $i$ and sample problems in the ratio $1 - s_i$ so that problems with lower success rates get a higher sampling probability. This focuses the model's efforts on its weakest areas, thereby accelerating learning and improving overall performance.### Code, Math and Visual Data

Due to the compliance restrictions of crawlers, many coding questions obtained online do not have test examples. To this end, the author uses [CYaRon1](https://github.com/luogu-dev/cyaron) to generate test cases to calculate reward. Of course, this requires many strict assumptions, such as no special judgments are required, and there are available ground truth solutions for these problems, so that these solutions can be used to generate higher quality test cases.

As for mathematics, one of the challenges in evaluating mathematical problems is that different expressions may represent the same answer. For example, $a^2 - 4$ and $(a + 2)(a - 2)$ may both be valid solutions to the same problem. In order to improve the scoring accuracy of the reward model, the authors tried two methods:

1. Classic RM: Referring to InstructGPT, the author implemented a value-head based reward model and collected about 800,000 data points for fine-tuning. The model ultimately takes `{question, answer, reference answer}` as input and outputs a scalar indicating whether the response is correct.
2. Chain-of-Thought RM: Some recent work has proven that reward models incorporating CoT are significantly more effective on tasks that require detailed standards. The authors went on to collect about 800,000 datasets with CoT to fine-tune the reward model. Based on the same input as Classic RM, the CoT model explicitly generates a step-by-step reasoning process and finally provides the final reward in JSON format. In the author's case study, Classic RM achieved an accuracy of approximately 84.4, while CoT RM achieved an accuracy of 98.5. Ultimately, they adopted CoT RM to ensure more accurate feedback.

Finally, Vision RL data mainly comes from three major categories: Real-world data, Synthetic visual reasoning data and Text-rendered data. Among them, the approach of text-rendered data is quite interesting, converting text content into images, which specifically emphasizes the model's ability to process text-dense images.

### Long2Short training

The inference overhead of the Long-CoT model is significantly greater, and the authors point out a variety of ways to migrate Long CoT capabilities to Short CoT. For example, model merging, shortest rejection sampling, DPO and Long2Short RL. Here I will first share the most violent method in terms of experience - model merging. It's very simple. Just average the parameters of the long cot model and the short cot model to get a better short cot model. Furthermore, shortest rejection sampling is to select the shortest and correct answer as the final sample from all correct samples each time. Of course, the other side of rejection sampling is DPO, which can use short wrong answers and correct long answers as negative samples to construct pairewise preference data. Likewise, simple rejection sampling can be widely used for math and coding problems because rule-based verification is more accurate than humans themselves. In the SFT stage, the authors also used rejection sampling to expand the data set.## RL Infra

This is the part that interests me the most. The authors focus on the idea of partial rollout in their RL system, which is the most important innovation.

### Partial Rollout

[This part comes from the proposal written by Yuzhen Zhou and I, so I moved it directly over. The writing is very specific]

**Problem description**

As we all know, in large-scale industrial RLHF systems, the inference phase (rollout) accounts for more than half of the overall process overhead. Considering a multi-task (mutli-tasks) PPO training, there are significant differences in decode length between different tasks; even within the same task, the decode length is also unbalanced.

Specifically, in the current rollout stage, the rollout engine mainly uses data parllesim. Each rollout worker is responsible for a part of the sampling tasks, and each maintains and completes all its own requests. The number of requests that need to be processed on each worker is similar, but the decode length difference between requests is significant. However, after requests are sent to each worker by the higher-level DP Manager, these requests will no longer be exchanged between workers. For example, if 100,000 prompts are allocated to 8 workers, each worker can process 12,500 prompts on average. This completely separated structure means that once there is a "slow task" within a certain shard, the entire training will be held back by this shard, and GPU resources cannot be fully utilized. As a result, the imbalance of decode length directly leads to the possibility of serious long-tail blocking problems in the rollout stage: under the premise of strict on-policy - the tracjories used for the current iteration training must be obtained by the policy model rollout of the current iteration - some requests complete the rollout quickly, while others take a long time, causing the pipeline of the entire rollout stage to be blocked by workers processing slow tasks, and resource utilization is significantly reduced. And as the amount of data for multi-task training increases, this blocking becomes more and more significant.

What's more serious is that the routing strategy adopted by the current dp manager is mostly based on prefix maximum, that is, the requests sent to each worker should have shared prefixes with each other as much as possible. This prefix maximum routing policy will assign tasks with similar prefix to the same worker, and tasks with similar prefix to long decode tasks are more likely to be long decode tasks. Although prefix maximum saves prefill overhead, it may send a large number of long decode requests to the same worker, which aggravates the situation we described earlier.To sum up, the imbalance of the existing Rollout process comes from the following aspects:

- Rollout synchronization: Due to strict on-policy requirements, the training phase can only be entered uniformly after all rollout requests are completed;
- Long-tail tasks slow down the overall situation: some computing resources are left waiting after completing short tasks, which is a waste of resources;
- Task isolation under DP settings makes it impossible to schedule waiting queues across workers;
- The mainstream routing policy ignores the impact of decode-heavy tasks while saving prefill overhead.

**Feasible solution Partial Rollout**

In order to solve the above problems, an optimization solution that has attracted widespread attention recently has emerged - Partial Rollout. The core idea of ​​this strategy is to sacrifice on-policy requirements to a certain extent. Instead of waiting for all prompts to complete inference, the sampling volume is increased, the parts that have completed rollout are selected for training first, and the remaining unfinished samples are deferred for processing.

Give a simple example:

- Each round of training only requires 128 samples, but 512 requests are started for inference at the same time;
- When 128 prompt rollouts are completed, the training process will be entered immediately;
- Prompt for the remaining 384 unfinished rollouts:
  - Continue to use the current policy to complete asynchronously (if training and inference do not interfere with each other);
  - Or abort and cache the current build state, resume in subsequent iterations, continue inference or start over.

**Strategic Tradeoffs**

Using Partial Rollout requires considering a policy model selection issue: whether these samples should continue to use the old model when they were initially started (old policy) to continue inference, or start over using the currently updated model (new policy). No matter what specific completion strategy is adopted, once it is allowed to train the completed part first and then process the remaining unfinished tasks, it will inevitably introduce strategic inconsistency in the training data - that is, the training data is no longer strictly derived from the latest policy, and may be mixed with some trajectories generated by old policies. This inconsistency is not an accident, but a necessity brought about by the Partial Rollout design, which is an active compromise for training efficiency. We need to make a trade-off between (non-)strict on-policy, training effect, and inference throughput.

Returning to the partial rollout itself, for samples that have not been sampled in the current iteration, if the model continues to complete the rollout after using parameter updates, then the generation process of this part of the data will span multiple policy stages and will no longer be complete "on-policy" training data. This saves time and computing power and improves overall training efficiency, but the effect is not strictly guaranteed. If you discard the unfinished samples of rollout and insist on using the latest policy to generate all training data from scratch, although the rigor of training is guaranteed, the inference overhead of previously unused samples will be wasted, affecting resource utilization.The core problem of Partial Rollout cannot be avoided. It must make a trade-off between training efficiency and policy consistency: it significantly improves the training throughput and reduces resource waste at the expense of part of the policy freshness. Although there is currently a lack of clear theoretical evidence to prove its negative impact, we should face up to its potential impact on policy learning and make a rational trade-off between engineering efficiency and algorithm rigor.

**What if you want to maintain strictly on policy? **

In fact, gradient accumulation provides a plausible reference. If we have 1024 data, divide it into 1024 / 4 = 256 data per batch. Each iteration will completely calculate the loss and update the gradient for the entire batch, but the model parameters will not be updated until 4 batches (256 * 4) are calculated. This can indeed ensure on policy. However, this method sounds feasible at first glance, but I personally think it doesn’t make much sense in practice:

It is very intuitive. Gradient accumulation solves the problem of OOM when using a large number of [complete] requests at one time, but patial rollout tries to solve the problem of getting [complete] requests that takes a very long time. Even if gradient accumulation is used, it will still take a lot of time to get [complete] trajcotries from scratch in one iteration. In addition, organizing a large batch size in multiple times is definitely not as efficient as using a large batch size all at once when circumstances permit. Because the same data has the same update scale (the effect is the same), gradient accumulation also increases the overhead of frequent switching between Rollout and Training.

Long context RL also has other system-side problems: in the deployment process of the existing rollout engine, the context length needs to be specified before the server is started because it needs to be directly compiled into the calculation graph. If a particularly long decoding request occurs during the inference process and exceeds the currently set context length, you can only restart the engine and reset a longer context length. We want to find better solutions to reduce the frequency of such restarts from the system and algorithm levels, or reduce the overhead of restarts.

**Summary**

In general, Partial Rollout provides a flexible and efficient multi-task RL training method to reduce the huge overhead of the decode phase in long context RL. It improves resource utilization and training rhythm, and is an effective means to deal with the long-tail problem of multi-task reasoning. However, the core cost of this strategy is the introduction of the inevitable off-policy training phenomenon, which is difficult to completely avoid for all similar asynchronous optimization designs in a multi-task environment. Therefore, how to mitigate the impact of policy drift on the performance of the final model is a problem we need to study.### RL framework

Back on k1.5, the author built a large-scale RL system based on megatron and vllm. In this industrial-level practice, their concerns are indeed very different from ours. For example, they will pay attention to the startup interval between the rollout stage and the training stage. It takes 1 minute from training to rollout, and 10 seconds in reverse. In addition, it also involves a variable that I have never considered-checkpoint engine. Simply put, as the rollout progresses, the longer the trajactoris length is required, the context length set by the rollout engine may not be large enough. The current approach is to repeatedly kill and relaunch the new rollout engine, which requires saving ckpt and minimizing restart overhead.

In terms of code execution, they use crun instead of dokcer, which is intuitively much faster.

## Ablation

- Short inference for large models and long inference for small models: Small models can use long inference to compete with large models, but overall, the token efficiency of large models is better. Large model with short context and small model with long context are currently the same solutions.
- Contrary to the ReST method, introducing negative gradient in training can significantly enhance the efficiency of the model in generating long cots.
- Course study (from easy to difficult) brings significant improvement.