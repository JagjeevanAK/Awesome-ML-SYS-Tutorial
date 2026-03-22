#OnlineDPO in TRL

## **Core components of OnlineDPO**

1. **Policy Model**: The trained model
2. **Reference Model**: Fixed baseline model, usually a frozen copy of the Policy Model
3. Evaluate components
    
    (Choose one of the two):
    
    - **Reward Model**: Scoring model, scores each generated result
    - **Judge**: Comparator, compares two generated results and chooses the better one

## **Calculation core formula**

The core of OnlineDPO is to maximize the probability ratio of selected responses relative to rejected responses. Its loss function is:

**Sigmoid loss**:

$\mathcal{L}{\text{DPO}}(\theta) = -\mathbb{E}{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \cdot \left( \log \frac{p\theta(y_w|x)}{p{\text{ref}}(y_w|x)} - \log \frac{p\theta(y_l|x)}{p{\text{ref}}(y_l|x)} \right) \right) \right]$

**IPO losses**:

$\mathcal{L}{\text{IPO}}(\theta) = \mathbb{E}{(x,y_w,y_l) \sim \mathcal{D}} \left[ \left( \log \frac{p\theta(y_w|x)/p{\text{ref}}(y_w|x)}{p\theta(y_l|x)/p{\text{ref}}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]$

Among them:

- $p_\theta(y|x)$ is the probability that the policy model generates a reply $y$ for a given prompt $x$
- $p_{\text{ref}}(y|x)$ is the corresponding probability of the reference model
- $\beta$ is a parameter that controls the strength of KL constraints
- $y_w$ and $y_l$ are selected and rejected replies respectively

## **Training steps**

### 1. Generation phase
```py
# Generate two different replies for each prompt
prompts = inputs["prompt"] # Shape: [batch_size]
batch_size = len(prompts)

# Use vLLM or standard generation
if use_vllm:
 prompt_ids, prompt_mask, completion_ids, completion_mask = _generate_vllm(model, prompts)
else:
 prompt_ids, prompt_mask, completion_ids, completion_mask = _generate(model, prompts)
```
At this stage:

- Extract tips from input batches
- Generate two different responses for each prompt (sampled twice)
- Check which replies contain closing token (EOS token)

### 2. Calculate model probability
```py
# Calculate the log probability of the policy model
logprobs = _forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)

# Calculate the log probability of the reference model (no gradient)
with torch.no_grad():
 If ref_model is not None:
 ref_logprobs = _forward(ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
 else: # In case of PEFT, just disable the adapter
 With model.disable_adapter():
 ref_logprobs = _forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
```
At this stage:

- Calculate the log probability of the policy model generating a reply
- Calculate the log probability of the same response from the reference model
- If using PEFT, you can obtain the probabilities of the reference model by disabling the adapter

### 3. Evaluate the generated results

```py
#Decode the generated reply
completions = processing_class.batch_decode(completion_ids, skip_special_tokens=True)

if judge is not None:
 # Use the judger for comparative evaluation
 Ranks = judge.judge(prompts, list(zip(completions[:batch_size], completions[batch_size:])))
 mask = torch.tensor([rank == 0 for rank in ranks], device=device)
else:
 # Use reward model for scoring
 scores = reward_model(prompt_completion_ids).scores

 # Handle replies that do not contain EOS (optionally lower their scores)
 If missing_eos_penalty is not None:
 scores[~contain_eos_token] -= missing_eos_penalty

 # Split scores and compare
 first_half, second_half = scores.split(batch_size)
 mask = first_half >= second_half
```
At this stage:

- Decode the generated token ID back to text
- Use discriminators or reward models to evaluate the quality of generated results
- Determine for each pair of responses which one is better (selected vs. rejected)

### 4. Organize data and calculate losses
```py
# Get the index of selected and rejected replies
batch_range = torch.arange(batch_size, device=device)
chosen_indices = batch_range + (~mask * batch_size)
rejected_indices = batch_range + (mask * batch_size)

# Get the logarithmic probability of being selected and rejected reply
chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)

# Calculate the log probability ratio
pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

# Logits required to calculate DPO loss
logits = pi_logratios - ref_logratios

# Calculate the loss based on the specified loss type
if loss_type == "sigmoid":
 losses = -F.logsigmoid(beta * logits)
elif loss_type == "ipo":
 losses = (logits - 1 / (2 * beta)) ** 2

loss = losses.mean()
```
At this stage:

- Identify which pair of responses was selected and which was rejected
- Calculate the log-probability ratio between the strategy model and the reference model
- Apply DPO loss function (sigmoid or IPO)

### 5. Update model

```py
# Perform backpropagation
if n_gpu > 1:
 Loss = loss.mean() # Average loss on multiple GPUs

accelerator.backward(loss, **kwargs)

# Return loss
return loss.detach() / args.gradient_accumulation_steps
```
At this stage:

- Perform backpropagation to calculate gradients
- Update model parameters by optimizer

##