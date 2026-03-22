#Tiny-LLM Day 1

It took me a while to configure the environment, because my new mac found that in order to cooperate with mlx, I need to install XCode. After a long time, I couldn't get around it, so I installed it. Then only python versions 3.10 to 3.12 can be used.

The overall experience feels like the documentation is relatively fly bitch. I wonder if it’s because Mr. Chi has done too many teaching assistants 🤣

Two functions are implemented here, and there are three functions in total. Let’s briefly summarize:

## Scaled Dot-Product Attention```python
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = key.shape[-1]
    scale = scale or 1.0 / sqrt(d_k)
    
    # Use multiplication instead of division
    score = (query @ mx.swapaxes(key, -2, -1)) * scale # Note that * is used here instead of /
    if mask is not None:
        score += mask
    
    attn = softmax(score, axis=-1) @ value
    return attn
```This function is naturally relatively simple. Let’s go through it quickly:

1. The attn formula we generally use is $$ \text{attn} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) V $$. Among them, dimensional information deserves special attention. The dimensions in the formula are all for two-dimensional tensors, but the dimension of the question input is `(N..., seq_len, d_model)`, which means that `N...` represents the batch dimensions of any situation. In fact, only the last two dimensions are used for matrix transposition. Therefore, using `key.T` directly will inevitably lead to failure. This transposes the last two dimensions, which is our `(seq_len, d_model)`.
2. This mask is not an attn mask, but a mask of attn score, which needs to be added to the attn score.
3. Use multiplication to complete scaled dot-product. This place tortured me for half an hour. In fact, we can write `(QK^T) / sqrt(d_k)`, but the numerical accuracy is not acceptable. First of all, converting general division into multiplying by the reciprocal is a very common method to stabilize numerical accuracy; in addition, large-scale matrix division is equivalent to losing precision for each element, while multiplying by the reciprocal will only lose the reciprocal calculation.
4. Softmax needs to have dimensions, but I think it is enough to remember.

## SimpleMultiHeadAttention

This question continues to deepen our understanding of dimensions. The title explains multiple dimension conversions, and I directly marked them in the code:```python
class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_dim = hidden_size // num_heads
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.scale = 1 / sqrt(self.hidden_dim) # hidden dim is for each head, not # the total hidden size
        assert wq.shape == (hidden_size, hidden_size)
        assert wk.shape == (hidden_size, hidden_size)
        assert wv.shape == (hidden_size, hidden_size) # wq wk wv are all (hidden_size, hidden_size) square matrices
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape # N is batch size, L is sequence length
        # The dimensions of Query are N, L, hidden_size (or embedding size, which is the same as hidden_size)

        # Here we directly use the written linear dot product. The actual execution of linear is y = xA^T + b
        # There is a conventional difference between the implementation of the linear layer in the deep learning framework and the way of writing mathematical formulas:
        # We are used to writing XW in mathematical formulas, but in actual code implementation, the weight matrix W is usually stored with transposition, that is to say, what is stored in the frame is actually W^T
        proejct_q = (
            linear(query, self.wq) # Split the hidden_size of the last dimension into num_heads hidden_dim to get (N, L, num_heads, hidden_dim)
            .reshape(N, L, self.num_heads, self.hidden_dim) # Also transpose, advance num heads
``````python
            .transpose(0, 2, 1, 3) # (N, num_heads, L, hidden_dim)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.hidden_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.hidden_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = scaled_dot_product_attention_simple(
            proejct_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        ) # (N, num_heads, L, hidden_dim)
        # First transpose num heads back, then reshape and merge them back to (N, L, hidden_size)
        attn = attn.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_dim * self.num_heads)
        return linear(attn, self.wo)
```