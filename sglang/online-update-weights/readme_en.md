# Online Update Weights

As described in [code-walk-through](../code-walk-through/readme.md), in order to achieve the integration of SGLang and OpenRLHF,
We need to add an `online_update_weights` interface to SGLang, which is different from the previous `update_weights`. The previous `update_weights` reads new weights from disk, while `online_update_weights` broadcasts new weights directly from the training engine through nccl.

## Existing `update_weights`

To add the same `online_update_weights` interface to every place where there is `update_weights` now. So here are some important `update_weights` interfaces.

### `ModelRunner`

`update_weights` in `sglang/srt/model_excutor/model_runner.py`, this function is as follows:

<details>
<summary>Code</summary>

```python
    def update_weights(self, model_path: str, load_format: str):
        """Update weights in-place."""
        from vllm.model_executor.model_loader.loader import (
            DefaultModelLoader,
            device_loading_context,
            get_model_loader,
        )
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype

        logger.info(
            f"Update weights begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)

        try:
            # TODO: Use a better method to check this
            vllm_model_config = VllmModelConfig(
                model=model_path,
                quantization=self.server_args.quantization,
                tokenizer=None,
                tokenizer_mode=None,
                trust_remote_code=self.server_args.trust_remote_code,
                dtype=self.server_args.dtype,
                seed=self.server_args.random_seed,
                skip_tokenizer_init=True,
            )
        except Exception as e:
            message = f"Failed to load model config: {e}."
            return False, message

        load_config = LoadConfig(load_format=load_format)

        # Only support vllm DefaultModelLoader for now
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source(
                    config.model,
                    revision=config.revision,
                    fall_back_to_pt=getattr(
                        self.model, "fall_back_to_pt_during_load", True
                    ),
                )
            )
            return iter

        def model_load_weights(model, iter):
            model.load_weights(iter)
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
            return model

        with set_default_torch_dtype(vllm_model_config.dtype):
            try:
                iter = get_weight_iter(vllm_model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.vllm_model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.vllm_model_config = vllm_model_config
        self.load_config = load_config
        self.model_config.path = model_path

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."
```
</details>In fact, it did two things:

- **Weight Loading**: Use vllm's `DefaultModelLoader` to load new model weights from the specified `model_path`

- **Configuration Update**: Update related model configuration information, including `server_args`, `model_config`

1. Create `VllmModelConfig` configuration
2. Get `ModelLoader`
3. Get the weight iterator through `get_weight_iter`
4. Use `model_load_weights` to load weights
5. Update related configurations

### `TpModelWorker`

In fact, `update_weights` of `TpModelWorker` represents a large class of `update_weights` interfaces, which are called layer by layer until `update_weights` of `ModelRunner`. It is worth mentioning that in [code-walk-through](../code-walk-through/readme.md#tpmodelworker), we mentioned that SGLang’s `TpModelWorker` and `ModelRunner` are jointly responsible for the `Worker` function of vllm. That is:

- `TpModelWorker`: Responsible for initializing the model and distributed environment, managing the memory pool, performing forward propagation of the model, classification processing embedding and generation tasks.
- `ModelRunner`: Responsible for actually performing model reasoning and providing an interface for `TpModelWorker` to call.

Therefore, OpenRLHF adds two interfaces related to `update_weights` based on vllm's `Worker`:

<details>
<summary>Code</summary>

```python
import importlib
import inspect

import torch
from vllm.worker.worker import Worker

from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WorkerWrap(Worker):
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl"):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #torch.cuda.empty_cache()`
```

Correspondingly, `online_update_weights` also needs to be added to `ModelRunner`.

</details>

### `Runtime`

Specifically, it refers to the `Runtime` class in `sglang/srt/server.py`, which is directly linked to the `Runtime` part in the previous analysis [../code-walk-through/readme.md](../code-walk-through/readme.md#runtime).


It is clear here that the `update_weights` request of `app` will be sent downward layer by layer, first sent to `TokenizerManager`, and then the `update_weights` of the actual `ModelRunner` will be called downward.