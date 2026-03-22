# RayPlacementGroup

This section details how slime orchestrates GPU resources on Ray, including:
- How to create and rearrange Placement Groups (PGs) to achieve stable GPU ordering
- How to schedule training Actors and Rollout Engine on PG
- Two deployment modes: colocate and dis-agg

---

## High Level Concepts

- [`Ray Placement Group`](https://github.com/THUDM/slime/blob/main/slime/ray/placement_group.py): Reserve a group of bundles (each containing 1GPU+1CPU) in the cluster, and fix subsequent actors to these bundles to achieve controllable and stable resource placement.
- [`RayTrainGroup`](https://github.com/THUDM/slime/tree/main/slime/ray/actor_group.py): Manager of "isomorphic actor group" on the training side.
  - Create training actors for each rank in stable order and provide concurrent init/train/eval/save/update/offload interfaces.
  - `self._actor_handlers`: Saves a list of all training actors with a length equal to world_size; all subsequent concurrent operations are mapping calls to this list. Created by [`_allocate_gpus_for_actor`](https://github.com/THUDM/slime/blob/main/slime/ray/actor_group.py#L67-L81).
  - Details will be placed in Part 3.
- [`RolloutManager`](https://github.com/THUDM/slime/tree/main/slime/ray/rollout.py): Inference/data orchestrator, responsible for creating Rollout Engine, Data Buffer, Lock and Router; its details are in Part 2.
- [`InfoActor`](): Temporarily detect actors, used to learn "the actual location of the bundle (node_ip, gpu_id)", thereby stably sorting the bundles.

---

## Core entrance and overall process
The entrance is located in `create_placement_groups` of [`Ray Placement Group`](https://github.com/THUDM/slime/blob/main/slime/ray/placement_group.py)
<details>
<summary>create_placement_groups</summary>

```py
#1:121:https://github.com/THUDM/slime/tree/main/slime/ray/placement_group.py
def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    print(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices),
        "rollout": (pg, rollout_pg_reordered_bundle_indices),
    }
```
</details>


Key points:
- Calculate the total number of GPUs required for this training `num_gpus`.
- Create a PG containing `num_gpus` bundles, each bundle requires `{"GPU": 1, "CPU": 1}`.
- Get a "reordered bundle index list" to ensure stable cross-node/GPU ordering.
- Divide the PG index between the training Actor and the Rollout engine according to `rollout_offset`.

---

## Stable Bundle reordering: sorting by node and GPU order

After creating PG, slime uses a temporary `InfoActor` to run once on each bundle to detect the `(Node IP, GPU ID)` actually assigned to the bundle, and then sort by "Node IP Numerical" and "GPU ID" to obtain a stable sequence.
<details>
<summary>`InfoActor` and `sort_key`</summary>

```py
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)
```
</details>

Key points of strategy for `sort_key`:
- Prioritize trying to parse `node_identifier` into an IPv4 address, convert it into 4 integers and sort them accordingly;
- If it is not an IP, try DNS resolution; if it still fails, it will degrade to sorting by the ASCII sequence of host name characters;
- Within the same node, sort by `gpu_id` in ascending order.

This can achieve a stable bundle order across multiple machines and avoid unstable mapping of "training rank to GPU" and "rollout shard to GPU" due to Ray's internal scheduling differences.

---

## Colocate vs Dis-agg

### Colocate (mixed cloth on the same machine)

Scenario: Training Actor and Rollout engine share the same batch of GPU resources (for example, 8 cards run both training and inference).

- Calculation method: `num_gpus = actor_num_nodes * actor_num_gpus_per_node`; `rollout_offset = 0`.
- means that the Rollout's bundle index slice comes from the same source as the Actor (i.e. all bundles of the same PG).
- Resource sharing is “proportionally divided” at the scheduling level:
  - The default value for training Actor is `num_gpus_per_actor = 0.8` (one Actor + several lightweight processes can coexist on one card).
  - The Rollout engine defaults to `num_gpus_per_engine = 0.2` (see below), allowing inference and training to coexist on the same card.

Applicable to: small-scale single/multiple machines, saving the number of nodes, and giving priority to total throughput, but you need to pay attention to the competition between inference and training for video memory and computing power.

### Dis-agg (training/inference separation)

Scenario: Actor training and Rollout engine use separate GPU pools (for example, training takes up 8 cards and rollout takes up 4 cards).

- Calculation method: `num_gpus = actor_total + rollout_total`, `rollout_offset = actor_total`.
- The PG is still one, but the first `actor_total` stable bundles are assigned to the training Actor, and the last `rollout_total` are assigned to the Rollout engine.
- Completely avoid resource contention, the inference service can be more stable; the price is the need for more GPUs.
- *async-train is possible in this case*

---

## Train vs Rollout

- Train (RayTrainGroup):
  - Binding order: Use "stable rearranged bundle index" to bind in order by rank; rank0 returns `MASTER_ADDR/PORT`.
  - Resource ratio: `num_gpus_per_actor` (default 0.8) allows sharing with the same card as rollout (when co-locate).
  - Concurrency management: Batch concurrent init/train/eval/save/update through `self._actor_handlers`.
  - Refer to the code snippet in "Overview of Components and Responsibilities" above for creating a loop.- Rollout (create_rollout_engines):
  - Binding strategy: also use rearrangement index; default `num_gpus=0.2` per engine, `placement_group_capture_child_tasks=True` makes child tasks also subject to PG constraints.
  - Cross-machine consistency: allocate service/NCCL/distributed initialization/DP-attention ports to multi-node engines to ensure that nodes within the same engine share `dist_init_addr`.
  - See Part 2 for detailed implementation.

---

## Port allocation and multi-machine consistency

Under multi-node/multi-card conditions, `create_rollout_engines` will use `RayActor._get_current_node_ip_and_free_port` to find a continuous available port on the target node, and spread the `dist_init_addr` of Node 0 to other nodes of the same engine to ensure cross-machine process group consistency.
<details>
<summary>`RayActor._get_current_node_ip_and_free_port`</summary>

```python
def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
    address = ray._private.services.get_node_ip_address()
    address = address.strip("[]")
    port = start_port
    while not all(is_port_available(port + i) for i in range(consecutive)):
        port += 1
    return address, port
```
</details>

---

## Choose colocate or dis-agg?

- Select colocate when:
  - Limited resources or simplified deployment are prioritized;
  - Be able to accept the resource contention and performance fluctuations caused by inference and training on the same card;
  - The Rollout engine is lighter and less sensitive to inference throughput.

- Select dis-agg when:
  - Pursue stable inference latency/throughput, or the inference load is heavy;
  - Resources are sufficient and it is expected that training and inference will not interfere with each other;
  - Resource pools for different businesses need to be isolated;
  - Want to do aysnc training.

---

## Summary

- Through "InfoActor detection + stable sorting", slime obtains stable bundle order across multiple machines;
- Training and Rollout share or separate resources, switching between colocate and dis-agg modes;
- Ports and distributed addresses are detected locally by the node where the engine is located, ensuring cross-machine consistency and reproducible deployment.

---

## FAQ: Why bind InfoActor? Why do you need sort_key? Can it be done directly in the order of allocation?

- Reasons for binding InfoActor:
  - **Detect the actual landing point**: The bundle index of PG is not equal to the physical (node, gpu); the real landing point can only be known by scheduling the task to the bundle.
  - **Get GPU number**: Only when `InfoActor` is run with `num_gpus=1`, `ray.get_gpu_ids()` will return the clear local GPU ID.
  - **Consistent with the follow-up**: Training/rollout will also use the same `placement_group_bundle_index` to bind to these bundles. The mapping should be first explored to facilitate stable placement and verification.

- Reasons why sort_key is needed:
  - **Ray does not guarantee the fixed order of bundle→topology**: PG internal resource mapping may change under different running/cluster states.
  - **Training and communication require stable order**: It is hoped that cross-nodes will be in ascending order by IP and within nodes will be in ascending order by GPU, so that NCCL topology, logs and troubleshooting can be consistent.
  - **sort_key strategy**: IP resolution → numerical sorting; if it fails, DNS will be used; if it fails again, it will be sorted by host name ASCII; within the node, it will be sorted by `gpu_id` in ascending order.

- Reasons why "allocation order" cannot be used directly:
  - **Uncontrollable and unstable**: As time/occupancy/health status changes, rank→GPU mapping drifts, making it difficult to reproduce experiments.
  - **Conflict with slicing/cross-machine strategy**: For example, dis-agg slicing and first card positioning of multiple cards per engine will introduce implicit mismatches due to order instability.