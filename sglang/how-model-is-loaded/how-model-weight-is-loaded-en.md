# SGLang model loading process

## Overview

SGLang's model loading process is implemented by the code in the `model_loader` folder.

The `get_model` function is defined in `__init__.py`, which is responsible for obtaining the corresponding `loader` according to `load_config` and calling `loader.load_model` to actually load the model:```cpp
def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    loader = get_model_loader(load_config)
    return loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )
```
The following takes `DefaultModelLoader` as an example to introduce in detail how to load a model from an open weight file into SGLang.

## DefaultModelLoader

1. **loader.load_model: model initialization (_initialize_model)**
   - `_initialize_model` calls `get_model_architecture` to get the model architecture according to `model_config`. `ModelRegistry.resolve_model_cls` will return the actual model class.
   - `_initialize_model` calls `_get_quantization_config` to obtain `quant_config` based on `model_config` and `load_config`:
     - `Linear` layers (like `ColumnParallelLinear` etc.) call `self.quant_method.create_weights` in `__init__` when actually initializing the model. For models without quantization methods, `quant_method` will be set to `UnquantizedLinearMethod`, and its `create_weights` method will create weight parameters with specified shapes and data types, and set meta-information such as input and output dimensions, and finally register them in the layer for subsequent use.
     - In the `forward` function of the `Linear` layer, `quant_method.apply` will be called for actual calculation.
2. **loader.load_model: Get weight iterator (_get_all_weights)**
   - Weights are divided into primary weights and secondary weights:
     - **Main Weights**: Load the `Source` and call `_get_weights_iterator` to actually load the weight parameters. `_get_weights_iterator` will load weights according to different weight formats (such as `.bin`, `safetensors`, `.pt`) and return a weight iterator.
     - **Secondary Weights**: No models currently use this feature.
3. **loader.load_model: Call model.load_weights** (take `qwen2` as an example)
   - `stacked_params_mapping` is used to map and load separate weight parameters in checkpoint (such as `q/k/v` or `gate/up`) to merged parameters in the model (such as `qkv_proj` or `gate_up_proj`).
     - `param_name`: The merged parameter name in the model.
     - `shard_name`: the individual original parameter name in checkpoint.
     - `shard_id`: The position of the original parameter after merging (such as "q", "k", "v" or 0, 1).
   - **Skip weights that don't need to be loaded**.
   - **Check if it is a stacking parameter**:
     - If so, replace `shared_name` (like `q_proj`) with `param_name` (like `qkv_proj`) in the name in the checkpoint (like `"model.layers.0.attn.q_proj.weight"`) to match `params_dict`.- **weight_loader**: Copies the loaded weight (`loaded_weight`) to the parameter (`param`) of the current model, and performs some special processing according to the parameter type and distributed configuration.
     - The `weight_loader` of each `linear` operator may be different. For specific differences, please refer to [linear.py file](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py).
4. **loader.load_model: After the weight loading is completed**, traverse each sub-module in the model. When quantization methods need to handle weights after loading (e.g. repacking, quantization, etc.), they expect the parameters to be on the global target device. This scope is useful when using CPU offloading, in which case SGLang will move the parameters to the device, perform `process_weights_after_loading` defined in the quantization method, and then move the parameters back to their original location.

------

## SGLang underlying operator (the following content is for reference only and still requires cross-validation)

- **`VocabParallelEmbedding`**: An embedding layer that supports vocabulary dimension parallelism (Tensor Parallelism) and LoRA dynamic expansion, designed for efficient and flexible management of vocabulary weights in multi-card inference and fine-tuning scenarios.

- **`ColumnParallelLinear`**: A linear layer is defined as `Y = XA + b`, where the matrix A is parallelized along the second dimension (column parallel), i.e. `A = [A_1, ..., A_p]`.

- **`RowParallelLinear`**: Supports splitting the weight matrix into linear layers of multiple cards by "rows" for parallel calculation of the linear part in `Y = X @ A + b` on multiple GPUs. Matrix A is parallel in the first dimension and X is parallel in the second dimension.  ```cpp
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
  ```
- **`QKVParallelLinear`**: QKV transformation linear layer used in the attention mechanism, responsible for the linear transformation of query, key and value vectors. The weight matrix is ​​concatenated along the output dimension and parallelized in the attention head dimension. When the number of key/value headers is less than the number of query headers (such as in multi-query or grouped query attention), the key and value headers are copied, and the query headers are split.

  - weight_loader：

    - `output_dim`: In which dimension does the current param (a weight tensor in the model) represent the "output dimension", that is to say - sharding or loading of slices is to be performed on this dimension.

    1. PART 1: Loading the merged QKV weights (fused tensor)

       1. Without `output_dim`, copy directly

       2. Manually define Q/K/V fragmentation information

          | name | offset | size |
          | ---- | --------- | ------------------------ |
          | q | Starting from 0 | num_heads × head_size |
          | k | immediately after q | num_kv_heads × head_size |
          | v | immediately after k | num_kv_heads × head_size |

       3. Finally call itself to load the single shard

    2. PART 2: If `loaded_shard_id` is `"q"`, `"k"`, or `"v"` - load a single shard.

- `MergedColumnParallelLinear`

  Packed linear layers with column parallelism. Similar to `ColumnParallelLinear`, but the weight matrix is ​​concatenated in the output dimension. When loading the weight matrix, different partitions are split separately.

  - weight_loader：

    - `output_dim`: In which dimension does the current param (a weight tensor in the model) represent the "output dimension", that is to say - sharding or loading of slices is to be performed on this dimension.

    1. Get the data of the model parameters (actual tensor) and some of its flags:
       - `output_dim`: along which dimension to shard (column parallel)
       - `is_metadata`: whether it is metadata (special format)
       - `needs_scalar_to_array`: Whether it is necessary to reshape the scalar into a vector (usually used for fused scale)
    2. If `loaded_shard_id` is `None`: load the whole block
       1. If there is no `output_dim`, copy directly (indicating that this weight does not need to be fragmented at all)
       2. If there is `output_dim` but no shard id, manually load the slicing call yourself. Traverse each output shard, divide this weight into multiple small shards, and call yourself respectively.
    3. When there is a shard_id (that is, a single shard is to be loaded): cut and load------

## BitsAndBytesModelLoader

When loading a model quantitatively using BitsAndBytes, `BitsAndBytesModelLoader` is called instead of `DefaultModelLoader`. The main difference is reflected in `loader.load_model`, which will be explained in detail below starting from `load_model`.

### loader.load_model (loader=BitsAndBytesModelLoader)

1. **Model initialization (`_initialize_model`)**.

2. **Load weights (`_load_weights`)**:

   - Verify whether the model supports BitsAndBytes quantization loading, and prepare the required parameters and states according to the configuration to load quantization weights.

   1. Verify parameters.

   2. **Get quantized weight iterator (`_get_quantized_weights_iterator`)**: Get the `qweight_iterator` used by `model.load_weights`, and also get the `QuantState` object `quant_state_dict`:

      - Get actual model weights via `_prepare_weights`.

      - `_quantized_4bit_generator`: actually loads and processes the 4-bit quantized weight file of the Hugging Face model.```cpp
        HuggingFace weight file →
          ├─ Collect all quantified metadata → temp_state_dict
          └─ Traverse the weight file →
                ├─ If quant_state exists → Construct QuantState object → Store in quant_state_dict
                └─ yield (param_name, weight_tensor)
        ```3. Call `model.load_weights`. This process is similar to `DefaultModelLoader`, except that `qweight_iterator` is passed in.

   4. Organize the quantization weights of shards according to unified parameter names and shard indexes to prepare for setting the quantization state.

   5. Set the quantization state, slice offset information, and runtime state required for 8-bit inference for each quantization parameter.

------

## BNB model weight```cpp
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.down_proj.weight.absmax
model.layers.0.mlp.down_proj.weight.nested_absmax
model.layers.0.mlp.down_proj.weight.nested_quant_map
model.layers.0.mlp.down_proj.weight.quant_map
model.layers.0.mlp.down_proj.weight.quant_state.bitsandbytes__nf4
...
```
| Weight name suffix | Function |
|-------------------------------- |------------------------------------------------ |
| `.absmax` | The maximum absolute value of the entire tensor, used for scaling recovery float |
| `.nested_absmax` | The maximum value calculated in chunks (fine-grained) |
| `.nested_quant_map` | Quantization information for each block |
| `.quant_map` | Mapping index table for decoding |
| `.quant_state.bitsandbytes__nf4` | Stores the actual NF4 encoded data |

------