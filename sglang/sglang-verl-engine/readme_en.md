# SGLang Verl Engine optimized analysis

## Interface summary

1. `update_weights_from_tensor` in `python/sglang/srt/entrypoints/engine.py````python
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be true
        to avoid duplicated operations such as clearing cache."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)
        )
```
Won't this interface be slow due to the serializer's serialization speed?

2. RPC in `python/sglang/srt/entrypoints/engine.py````python
    """
    Execute an RPC call on all scheduler processes.
    """

    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)
        self.send_to_rpc.send_pyobj(obj)
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        assert recv_req.success, recv_req.message

    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)
```
RPC (Remote Procedure Call) is a technology that implements inter-process communication in a distributed system. In this code, RPC is used to communicate between the main process and scheduler processes.

1. The `collective_rpc` method is used to send RPC requests to all scheduler processes;
2. Implement inter-process communication through ZMQ (message queue middleware):
   - `send_pyobj` sends a serialized Python object as a request
   - `recv_pyobj` receives response
3. The request contains:
   - `method`: the name of the remote method to be called
   - `parameters`: method parameters (passed in the form of kwargs)

As can be seen from the code context, this RPC mechanism is mainly used for:
- Save remote model (`save_remote_model`)
- Save sharded model (`save_sharded_model`)

This design allows the main process to control and coordinate the behavior of multiple scheduler processes and is an important component in a distributed inference system.

3. `DeviceMesh, DTensor` in `python/sglang/srt/entrypoints/verl_engine.py````python
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor

from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.server import Engine
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj


class VerlEngine:
    def __init__(
        self,
        device_mesh_cpu: DeviceMesh,
        nnodes: int = 1,
        **kwargs,
    ):
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._tp_size = device_mesh_cpu.size()
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = Engine(
                **kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes
            )
        else:
            self._engine = None

        dist.barrier(group=self._device_mesh_cpu.get_group())
```
DeviceMesh and DTensor are important components for distributed computing in PyTorch, mainly used for model parallelism and tensor parallel computing.

1. DeviceMesh:
- Is a logical device array used to manage device topology in distributed computing;
- Defines how computation is organized and coordinated across multiple devices such as GPUs
- In the code, `device_mesh_cpu` is used for:```python
self._tp_rank = device_mesh_cpu.get_local_rank() # Get the local ranking of the current process
self._tp_size = device_mesh_cpu.size() # Get the total number of devices
```2. DTensor (distributed tensor):
- is a distributed tensor type in PyTorch;
- Allows sharding a large tensor to multiple devices;
- Processing in code:

The naming of `device_mesh_cpu` reflects its actual use, as a control structure at the CPU level, used to coordinate process communication and task allocation in distributed systems, while specific GPU calculations are managed through other mechanisms.