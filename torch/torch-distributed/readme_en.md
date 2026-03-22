# Learning path

In order to write a `weight_update` interface for OpenRLHF, Mercy said to me, "You only need to learn `torch.dist`." When I heard it, "How do I remember that there is an interface of torch that calculates distance, called `torch.dist`?" Then he said, "It is actually `torch.distributed`." The whole room burst into laughter...

It doesn’t matter, I really want to learn `torch.distributed`:

1. Learn torch.distributed https://pytorch.org/docs/stable/distributed.html
2. How to create a communication group.
3. How to broadcast a tensor. 8 GPUs, 1 process per GPU, 8 processes. Broadcast a torch tensor from 1 GPU to the other 7 GPUs with torch.distributed (nccl backend).

After reading the learning objectives, I asked the guy next to my workstation how to learn `torch.distributed`. He said to read the source code, and I originally wanted to learn from the source code, but after taking a few glances, I gave up directly 😂. I checked online and found no good tutorials. It doesn't matter, I will take action! Ask claude and learn!

## `torch.distributed` Learning Thread

1. Basic concepts

- Process Group - the basic communication unit in distributed training

- Backend - focus on the NCCL backend because you need to use GPU communication

- rank - process number used to identify processes on different GPUs

- world size - total number of processes, in your case 8

2. Core API learning sequence

- Initialize distributed environment `init_process_group`

- Create custom distribution group `new_group`

## Mainstream communication interface

1. Point-to-point communication vs collective communication
2. `send` and `recv`
3. `all_reduce` and `all_gather`
4. `broadcast`
5. `scatter`

# `torch.distributed`

## Distributed computing in torch

I definitely don’t need to explain why distributed computing is needed 😂. `torch.distributed` is a module in PyTorch specially designed for distributed training, providing tools for communicating data and model parameters between multiple GPUs or nodes. Different from the traditional `torch` function, `torch.distributed` focuses on how to effectively coordinate and share data on multiple devices so that each device can work together in different training tasks. `torch.distributed` provides a communication interface, allowing users to implement parameter synchronization, gradient summary, broadcast and other operations in a multi-process environment, ensuring that all devices maintain the same model state in each round of training.In contrast, the ordinary `torch` function is designed based on a single process and a single device by default. Even in the case of multiple GPUs, ordinary PyTorch can only control one process to train models on multiple devices, but cannot support multiple processes cooperating on multiple devices. `torch.distributed` provides a high-level abstraction that allows users to easily manage multiple devices or nodes working together.

Based on distributed computing, distributed training and the distributed reasoning I am learning can be constructed. As far as training is concerned, there are at least two obvious categories:

1. **Data Parallelism**: This is the most common form of distributed training and is suitable for most deep learning tasks. When the entire model can be fully accommodated on each GPU, data parallelism distributes copies of the same model across multiple GPUs, each GPU responsible for processing a different part of the data set, and then aggregates the gradients and updates the model parameters through collective communication operations such as `all_reduce`.
- **Advantages**: Easy to implement, especially in areas such as image classification and NLP, which can be directly applied.
- **Implementation method**: Through `init_process_group()`, `all_reduce()` and other functions of `torch.distributed`, you can easily synchronize the gradient of each process and achieve data parallelism.

2. **Model Parallelism**: When the model scale is extremely large, the video memory of a single device is not enough to store the model parameters. In this case, the model can be split into different parts, and multiple GPUs are responsible for different parts of the model.
- **Advantages**: Can train large models whose memory exceeds the load of a single GPU.
- **Implementation method**: `torch.distributed` realizes data exchange between different modules of the model through point-to-point communication functions such as `send()` and `recv()`, thereby achieving model parallelism.

## Process Group

In a distributed system, the process group is a core communication unit. A process group organizes a group of existing processes together so that data can be exchanged between these processes through specific communication methods. When each process starts, you need to first use `torch.distributed.init_process_group` to initialize the distributed environment and add the process to the default global process group WORLD group. Afterwards, you can create a new process group through `torch.distributed.new_group` to organize specific processes together. Different process groups can use different communication methods, which can achieve a more flexible distribution strategy.

## `init_process_group`

Create a global process group and add processes to it. The name of this API is a bit confusing, because this command will be executed once in each process. It sounds like 8 global default process groups are started. In fact, what is done here is similar to the touch command. **The first process executed here is created and added to the global process group, and subsequent processes executed only need to be added. **<details>
<summary>init_process_group</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size):
    print(f"The process has been started: the rank of this process is {rank}")
    
    #Set the GPU used by the current process
    torch.cuda.set_device(rank)
    
    try:
        # Join process group
        print(f"Process {rank} is joining the process group...")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        print(f"Process {rank} has successfully joined the process group")
        
        # Verify identity
        assert rank == dist.get_rank()
        assert world_size == dist.get_world_size()
        
        # Prepare information about the current process
        process_info = (
            f"\nProcess {rank} information:\n"
            f"- Device: {torch.cuda.current_device()}\n"
            f"- GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}\n"
        )
        
        # Convert string to fixed length tensor
        max_len = 100 # Make sure it is long enough to hold the information
        process_info_tensor = torch.zeros(max_len, dtype=torch.int32, device='cuda')
        process_info_bytes = process_info.encode('utf-8')
        process_info_tensor[:len(process_info_bytes)] = torch.tensor([b for b in process_info_bytes], dtype=torch.int32)
        
        #Create a list of tensors to collect information about all processes
        gathered_tensors = [torch.zeros(max_len, dtype=torch.int32, device='cuda') for _ in range(world_size)]

        # Use all_gather to collect information about all processes
        dist.all_gather(gathered_tensors, process_info_tensor)


        if rank == 0:
            print("================ All process information ===============")
            for tensor in gathered_tensors:
                info_bytes = tensor.cpu().numpy().astype('uint8').tobytes()
                info_str = info_bytes.decode('utf-8', 'ignore').strip('\x00')
                print(info_str)
        
        # Create tensors and communicate
        tensor = torch.ones(1).cuda() * rank
        print(f"Original tensor value of process {rank}: {tensor.item()}")
        
        # All process synchronization points
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Final tensor value of process {rank}: {tensor.item()}")
    
    finally:
        dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = torch.cuda.device_count()
    print(f"Ready to start {world_size} processes...")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
    #! Equivalent to starting the process through the following code
    # processes = []
    # for rank in range(world_size):
    # p = mp.Process(target=init_process, args=(rank, world_size))
    #p.start()
    # processes.append(p)

    # # Equivalent to the effect of join=True
    # for p in processes:
    #p.join()

if __name__ == "__main__":
    main()
```

</details>

The core of this code is these three interfaces:

1. Add the process to the global process group

`dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)`


2. Use `all_gather` to collect device information of all processes

`dist.all_gather(gathered_tensors, process_info_tensor)`

Each process sends its own information to all other processes

3. Use `all_reduce` to sum tensors

`dist.all_reduce(tensor, op=dist.ReduceOp.SUM)`


## `new_group`

 Create a custom process group, the same as `init_process_group()`, create or join.


<details>
<summary>new_group</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size):
    try:
        # 1. Join the global process group
        torch.cuda.set_device(rank)
        if rank == 0:
            print(f"Ready to join the global process group...")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # 2. Create two custom process groups
        group1_ranks = list(range(world_size // 2))
        group2_ranks = list(range(world_size // 2, world_size))
        
        #Initialize the accumulated value to 0
        group1_sum = torch.zeros(1).cuda()
        group2_sum = torch.zeros(1).cuda()
        if rank == 0:
            print(f"Initialized accumulated value of group 1: {group1_sum.item()}")
            print(f"Initialized accumulated value of group 2: {group2_sum.item()}")
        
        group1 = dist.new_group(group1_ranks)
        group2 = dist.new_group(group2_ranks)
        
        # 3. Communicate within their respective groups
        tensor = torch.ones(1).cuda() * rank # The input value of each process is its rank
        if rank == 0:
            print(f"\nStart intra-group communication...")
        
        if rank == 0:
            print(f"Group1 performs all_reduce operation...")

        # Perform all_reduce in the corresponding group, and the accumulated results will be updated to the tensor.
        if rank in group1_ranks:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group1)
            group1_sum = tensor.clone() # Save the cumulative result of group1
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group2)
            group2_sum = tensor.clone() # Save the accumulated result of group2
        
        # Ensure that all processes can obtain the cumulative results of the two groups
        dist.all_reduce(group1_sum, op=dist.ReduceOp.MAX)
        dist.all_reduce(group2_sum, op=dist.ReduceOp.MAX)
        
        if rank == 0:
            print("\n================ Communication completed ==============")
            print(f"Group1 (ranks {group1_ranks}): The cumulative result is {group1_sum.item()}")
            print(f"Group2 (ranks {group2_ranks}): The cumulative result is {group2_sum.item()}")
    
    finally:
        dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = torch.cuda.device_count()
    print(f"Ready to start {world_size} processes...")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```
</details>


These codes are quite simple. What is more interesting is that the code for rank 0 has retained the accumulated results of the first group after passing `dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group1)`, but these two lines of code are still needed:```python
# Ensure that all processes can obtain the cumulative results of the two groups
dist.all_reduce(group1_sum, op=dist.ReduceOp.MAX)
dist.all_reduce(group2_sum, op=dist.ReduceOp.MAX)
```Because rank 0 is in group 1, `all_reduce(group1_sum)` and taking the maximum value have no effect on group1_sum. However, the group2_sum of rank 0 is still 0, and such an all_reduce is needed to accept the group2_sum of other ranks. Based on this, imagine simply changing `dist.ReduceOp.MAX` to `dist.ReduceOp.SUM`, and the result will be 4 times the previous one.

# Communication interface

There is an obvious need for communication between processes. What is interesting is that simple data parallelism requires complex `all_reduce, all_gather, broadcast`, while more complex model parallelism requires intuitively simpler `send, recv`. Let’s make a rough classification of these communication methods:

1. **Point-to-Point Communication**

Point-to-point communication is the most basic communication mode, which means that one process directly sends or receives data to another specific process. This mode is very flexible and suitable for scenarios that require precise control of the communication process.

- **send-receive mode**: In `torch.distributed`, this mode can be implemented through the `send()` and `recv()` interfaces. For example, `send(tensor, dst=1)` means that the process sends data to the process with rank 1, while `recv(tensor, src=0)` means receiving data from the process with rank 0. There is no doubt that this is blocking.

The advantage of point-to-point communication is that it is simple and intuitive, easy to understand and control; the disadvantage is that it can easily lead to complex code structures, especially when multiple processes need to send data to each other, deadlock or blocking problems may occur. Therefore, this method is more suitable for information exchange between two processes. It is suitable for scenarios that require precise control of data exchange between individual processes. It is usually used in system layer communication optimization or model sharding. For example, in gradient updates for model parallel training, point-to-point communication can be used for gradient summary.

2. **Collective Communication**

Collective communication is an advanced communication pattern typically used for data exchange between multiple processes. Set communication operations often involve all participating processes, so they are used very frequently in distributed deep learning.

- **Broadcast**: Broadcast is a communication operation that sends data from one source process to all other processes. In `torch.distributed`, this operation can be achieved through `broadcast(tensor, src=0)`, which broadcasts the data in the process with rank 0 to all other processes. The broadcast operation can ensure that all processes have the same data, and is suitable for scenarios where model parameters, initialization weights, etc. need to be shared. For example, in the initialization phase of distributed training, it is used to broadcast the model parameters of the main process to all other processes to ensure that training starts with the same initial parameters.
- **Reduction (Reduce and All-Reduce)**: The reduction operation is an operation that calculates data from multiple processes (such as summation, maximum value, etc.). There are two commonly used reduction operations, `reduce()`: one process (usually the main process) collects and merges data from all processes; `all_reduce()`: all processes get the merged data at the same time. For example, `all_reduce(tensor, op=ReduceOp.SUM)` will sum up in all processes and store the result in the `tensor` of each process. The reduction operation can effectively reduce the communication burden and is suitable for large-scale gradient aggregation or model weight update. For example, in distributed training, `all_reduce` is often used for gradient summation to ensure that the gradients in multiple processes remain consistent and achieve synchronous updates.- **Gather (Gather and All-Gather)**: The collection operation is an operation that collects data from multiple processes into one or more processes: `gather()`: Collects data from multiple processes into one process. `all_gather()`: All processes collect data of all processes. For example, `all_gather(gathered_tensors, tensor)` will collect `tensor` from all processes into the `gathered_tensors` list of each process. Collection operations facilitate subsequent analysis and processing of data from all processes. For example, when doing evaluation, you can use `all_gather` to summarize the intermediate results of each process.
- **Scatter**: The `scatter()` operation is to scatter the data of one process to multiple processes. For example, if the process with rank 0 has a list containing several sub-tensors, `scatter()` can allocate each sub-tensor in the list to other processes. Ideal for data distribution, spreading large datasets or model weights across multiple processes so that each process can process different chunks of data.

3. **Comparison of point-to-point and collective communication**

- **Flexibility**: Point-to-point communication is suitable for scenarios that require high-precision control communication, but is not suitable for large-scale communication because the code will become complex. Collective communication is more efficient and suitable for multi-process collaboration scenarios, especially in deep learning training.
- **Complexity**: Set communication simplifies common requirements such as data synchronization and gradient reduction, and can improve training speed and communication efficiency.

## `send` and `recv`


<details>
<summary>send and recv</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # Create initial data (only create meaningful data at rank 0)
        if rank == 0:
            tensor = torch.tensor([100, 200], dtype=torch.float32).cuda()
            print(f"\n=== initial state ===")
            print(f"Initial data of Rank 0: {tensor}")
            # Send data to rank 1
            dist.send(tensor, dst=1)
            print(f"Rank 0 has sent data to Rank 1")
            
        elif rank == 1:
            # rank 1 receives data from rank 0
            tensor = torch.zeros(2, dtype=torch.float32).cuda()
            dist.recv(tensor, src=0)
            print(f"Rank 1 received data: {tensor}")
            
            # Modify the data and send it to rank 2
            tensor = tensor * 2 # Double the data
            print(f"Rank 1 processed data: {tensor}")
            dist.send(tensor, dst=2)
            print(f"Rank 1 has sent data to Rank 2")
            
        elif rank == 2:
            # rank 2 receives data from rank 1
            tensor = torch.zeros(2, dtype=torch.float32).cuda()
            dist.recv(tensor, src=1)
            print(f"Rank 2 received data: {tensor}")
            print("\n=== Transfer completed ===")
            
    finally:
        dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = 3
    print(f"Ready to start {world_size} processes...")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```
</details>
Usage is very simple.

- `recv` needs to pre-allocate the tensor to receive data, and the size must match.
- Both `send` and `recv` are blocking operations. The sender will wait until the receiver completes reception, and the receiver will wait until the sender's data arrives.
- Each `send` must have a corresponding `recv`. Improper pairing will cause deadlock.

An example of improper use:

```python
# Error example - may lead to deadlock
if rank == 0:
    dist.send(tensor1, dst=1) # Wait for rank 1 to receive
    dist.recv(tensor2, src=1) # You can never wait because rank 1 is stuck in sending
elif rank == 1:
    dist.send(tensor2, dst=0) # Wait for rank 0 to receive
    dist.recv(tensor1, src=0) # You can never wait because rank 0 is stuck in sending

# Correct example
if rank == 0:
    dist.send(tensor1, dst=1)
    dist.recv(tensor2, src=1)
elif rank == 1:
    dist.recv(tensor1, src=0) #Receive first
    dist.send(tensor2, dst=0) # Send again
```

- The sending and receiving tensors must be on the same type of device (both on CPU or both on GPU).

- For simple collective communication, it is recommended to use specialized collective communication primitives: `all_reduce` replaces the summation of multiple `send/recv`, `all_gather` replaces the data collection of multiple `send/recv`, and `broadcast` replaces one-to-many sending.

## `isend` and `irecv`

- If you need non-blocking communication, you can use `isend/irecv`
- You can also use [dist.batch_isend_irecv](https://pytorch.org/docs/stable/distributed.html#torch.distributed.batch_isend_irecv) to fuse multiple P2P communication operations. This function will try to [fuse multiple NCCL kernel to improve throughput](https://github.com/pytorch/pytorch/issues/132852), and re-order the communication sequence to reduce the probability of deadlock.

<details>
<summary>isend and irecv</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time # Add time to demonstrate asynchronous effects

def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        if rank == 0:
            tensor = torch.tensor([100, 200], dtype=torch.float32).cuda()
            print(f"\n=== initial state ===")
            print(f"Initial data of Rank 0: {tensor}")
            
            # Asynchronously send data to rank 1
            print(f"Rank 0 ready to send data...")
            send_req = dist.isend(tensor, dst=1)
            print(f"Rank 0 starts asynchronous sending")
            
            # Simulate doing other work while waiting for sending to complete
            print(f"Rank 0 is processing other tasks...")
            time.sleep(1) # Simulate other computing tasks
            
            # Wait for sending to complete
            send_req.wait()
            print(f"Rank 0 confirmation sending completed")
            
        elif rank == 1:
            tensor = torch.zeros(2, dtype=torch.float32).cuda()
            print(f"Rank 1 ready to receive data...")
            
            # Asynchronously receive data from rank 0
            recv_req = dist.irecv(tensor, src=0)
            print(f"Rank 1 starts asynchronous reception")
            
            # Simulate doing other work while waiting for reception to complete
            print(f"Rank 1 is processing other tasks...")
            time.sleep(1) # Simulate other computing tasks
            
            # Wait for reception to complete
            recv_req.wait()
            print(f"Rank 1 reception completed, data is: {tensor}")
            
            # Process the data and send it to rank 2 asynchronously
            tensor = tensor * 2
            print(f"Rank 1 processed data: {tensor}")
            print(f"Rank 1 is preparing to send data to Rank 2...")
            send_req = dist.isend(tensor, dst=2)
            print(f"Rank 1 starts asynchronous sending")
            
            # Simulate doing other work while waiting for sending to complete
            print(f"Rank 1 is processing other tasks...")
            time.sleep(1) # Simulate other computing tasks
            
            send_req.wait()
            print(f"Rank 1 confirmation sending completed")
            
        elif rank == 2:
            tensor = torch.zeros(2, dtype=torch.float32).cuda()
            print(f"Rank 2 is ready to receive data...")
            
            # Receive data from rank 1 asynchronously
            recv_req = dist.irecv(tensor, src=1)
            print(f"Rank 2 starts asynchronous reception")
            
            # Simulate doing other work while waiting for reception to complete
            print(f"Rank 2 is processing other tasks...")
            time.sleep(1) # Simulate other computing tasks
            
            # Wait for reception to complete
            recv_req.wait()
            print(f"Rank 2 reception completed, final data is: {tensor}")
            print("\n=== Transfer completed ===")
            
    finally:
        dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = 3
    print(f"Ready to start {world_size} processes...")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```
</details>

- Do not modify the sending buffer (buffer) before the communication is completed. Do not use the receiving buffer before the communication is completed. You must wait for `wait()` to complete before you can safely operate the relevant data.
- Each asynchronous operation will occupy system resources, and `wait()` should be called in time to release resources.
- Avoid initiating too many unfinished asynchronous operations at the same time
- Asynchronous operations may fail in the background. `wait()` calls will expose errors in the communication process. It is recommended to use `try-finally` to ensure that resources are cleaned up correctly.

## `all_reduce` and `all_gather`

1. **Function Positioning**:
- `all_reduce`: perform reduction operations on the data of all processes, such as summation, maximum value, etc.
- `all_gather`: Collect the data of all processes, no operation, just simple merging

2. **Output results**:
- `all_reduce`: all processes get the same reduction result
- `all_gather`: all processes get a complete list containing the raw data of all processes

3. **Memory usage**:
- `all_reduce`: the output tensor size is the same as the input
- `all_gather`: The output tensor size is `world_size` times the input

4. **Applicable scenarios**:
- `all_reduce`: calculate distributed loss, gradient synchronization, calculate global statistics (such as accuracy)
- `all_gather`: obtain the original data of other processes, calculate distributed evaluation indicators, and summarize the intermediate results of different processes

5. **Communication efficiency**:

- `all_reduce` is usually more efficient than `all_gather`. If you only need to get the final summary result, `all_reduce` should be used first. The amount of data transmitted is smaller, and the tree structure can be used for reduction.


<details>
<summary>all_reduce and all_gather</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        #Create test tensor
        tensor = torch.tensor([rank * 10, rank * 10 + 1], dtype=torch.float32).cuda()
        
        # === all_gather example ===
        gathered = [torch.zeros(2, dtype=torch.float32).cuda() for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        
        if rank == 0:
            print("\n=== all_gather result ===")
            print(f"Original tensor (rank 0): {tensor}")
            print("Collect tensors of all processes:")
            for i, t in enumerate(gathered):
                print(f"data of rank {i}: {t.tolist()}")
        
        # === all_reduce example ===
        reduced_tensor = tensor.clone() # Create a copy for all_reduce
        if rank == 0:
            print(f"before all_reduce: {reduced_tensor}")

        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print("\n=== all_reduce result ===")
            print(f"Original tensor (rank 0): {tensor}")
            print(f"Reduced tensor (sum of all ranks): {reduced_tensor}")
            
    finally:
        dist.destroy_process_group()
        
def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = torch.cuda.device_count()
    print(f"Ready to start {world_size} processes...")
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```
</details>

In fact, `all_reduce` itself only supports limited operations, and some complex functions can be implemented through a combination of these operations, similar to implementing distributed `softmax`.


<details>
<summary>all_reduce implements softmax</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        #Create more complex test tensors
        tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.float32).cuda()
        
        if rank == 0:
            print(f"\nInitial tensor (rank {rank}): {tensor}")
            
        # 1. Use PREMUL_SUM to implement weighted sum
        weights = torch.tensor([0.3, 0.3, 0.4]).cuda()
        weighted = tensor * weights
        dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print(f"\n=== Weighted sum result ===")
            print(f"Weighted tensor: {weighted}")
            
        # 2. Implement the distributed version of softmax
        # Step one: Calculate the maximum value
        max_tensor = tensor.clone()
        if rank == 0:
            print(f"max_tensor before all_reduce: {max_tensor}")
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        
        if rank == 0:
            print(f"max_tensor after all_reduce: {max_tensor}")
        
        # Step 2: Calculate exp(x - max(x))
        exp_tensor = torch.exp(tensor - max_tensor)
        
        # Step 3: Calculate the denominator (sum of all exp)
        sum_exp = exp_tensor.clone()
        if rank == 0:
            print(f"sum_exp before all_reduce: {sum_exp}")
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print(f"sum_exp after all_reduce: {sum_exp}")
        
        # Step 4: Calculate the final softmax
        softmax_result = exp_tensor / sum_exp
        
        if rank == 0:
            print(f"\n=== Distributed Softmax result ===")
            print(f"Softmax result: {softmax_result}")
            
        # 3. Implement distributed version of L2 regularization
        # Step 1: Calculate the square
        squared = tensor ** 2
        
        # Step 2: Find the sum of the squares of all elements
        sum_squared = squared.clone()
        if rank == 0:
            print(f"sum_squared before all_reduce: {sum_squared}")
        dist.all_reduce(sum_squared, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print(f"sum_squared after all_reduce: {sum_squared}")
        
        # Step 3: Calculate the square root
        l2_norm = torch.sqrt(sum_squared)
        
        # Step 4: Regularization
        normalized = tensor / l2_norm
        
        if rank == 0:
            print(f"\n=== Distributed L2 regularization result ===")
            print(f"Regularization result: {normalized}")
            
    finally:
        dist.destroy_process_group()
        
def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = torch.cuda.device_count()
    print(f"Ready to start {world_size} processes...")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```
</details>

## `broadcast`

<details>
<summary>broadcast</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size):
    try:
        #Initialize process group
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        #Create data
        if rank == 0:
            data1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cuda()
            data2 = torch.zeros(3).cuda() # Used to receive the broadcast of rank 1
            print(f"Rank 0 initial data: data1={data1}, data2={data2}")
        elif rank == 1:
            data1 = torch.zeros(5).cuda() # Used to receive the broadcast of rank 0
            data2 = torch.tensor([10.0, 20.0, 30.0]).cuda()
            print(f"Rank 1 initial data: data1={data1}, data2={data2}")
        else:
            data1 = torch.zeros(5).cuda()
            data2 = torch.zeros(3).cuda()
            print(f"Rank {rank} initial data: data1={data1}, data2={data2}")
        
        # Execute the broadcast of rank 0 first
        dist.broadcast(data1, src=0)
        print(f"Rank {rank} after the first broadcast: data1={data1}")
        print(f"Rank {rank} after the first broadcast: data2={data2}")
        
        # Execute the broadcast of rank 1 again
        dist.broadcast(data2, src=1)
        print(f"Rank {rank} after the second broadcast: data1={data1}")
        print(f"Rank {rank} after the second broadcast: data2={data2}")

    finally:
        dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = torch.cuda.device_count()
    print(f"Ready to start {world_size} processes...\n")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```
</details>

The example is very simple:

1. `broadcast` broadcasts the tensor data of the source process src to the tensors of the same name in all other processes
2. The process receiving data must pre-allocate tensor space of the same size
3. The broadcast operation is blocking, and all processes need to execute this line of code before they can continue.
4. The data will be modified directly on the pre-allocated memory instead of creating a new tensor.

## `scatter`


<details>
<summary>scatter</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # === scatter example ===

        if rank == 0:
            # Create data to be distributed at rank 0
            # Prepare 2 numbers for each process
            scatter_list = [
                torch.tensor([i * 10, i * 10 + 1], dtype=torch.float32).cuda()
                for i in range(world_size)
            ]
            print("\n=== data before scatter ===")
            for i, tensor in enumerate(scatter_list):
                print(f"Data ready to be sent to rank {i}: {tensor.tolist()}")
        else:
            scatter_list = None

        # Prepare the tensor to receive data
        output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
        print(f"Rank {rank} initialization to receive data: {output_tensor.tolist()}")
        
        #Perform scatter operation
        dist.scatter(output_tensor, scatter_list, src=0)
        
        #Print the received data for each process
        print(f"Rank {rank} received data: {output_tensor.tolist()}")
            
    finally:
        dist.destroy_process_group()
        
def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    world_size = torch.cuda.device_count()
    print(f"Ready to start {world_size} processes...")
    
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```
</details>

- `scatter` is a one-to-many distribution operation, only the source process (here rank 0) needs to prepare complete data
- `scatter_list` of other processes must be set to None, this is a PyTorch requirement
- The data must be split according to the number of processes in advance, and each process gets one copy
- `scatter` operation is synchronous, all processes will wait here until communication is completed
- The source process (src=0) must be specified to indicate the process from which the data is distributed
- Each tensor in `scatter_list` must be the same size
- The total data size must be evenly divisible by the number of processes

- `scatter` is suitable for dividing large data sets into multiple processes for processing

- Compared to `broadcast`, `scatter` can save the memory usage of other processes

**`scatter` suitable for:**

1. Distribute different data batches during data parallel training
2. Shard large-scale data sets to multiple nodes for processing
3. Distribute model parameters in parameter server architecture

**Why is it said that `scatter` saves space compared to `broadcast`? **

Consider a total of 4 processes. It is necessary to send `[1000, 250]` dimensional data from rank 0 to ranks 1, 2, and 3. Then using `broadcast`, each card must have a data block of `[1000, 250]` size, and then slice each card. Using `scatter`, only rank 0 will have `[1000, 1000]`, and other ranks will have `[1000, 250]`.

# Postscript

Here are excerpts from some blogs that have inspired me.

Refer to Zhihu [[Original][Depth][PyTorch] The first article in the DDP series: Getting Started Tutorial](https://zhuanlan.zhihu.com/p/178402798).

There are several very important concepts. I will continue to ask claude to add:

## GIL

As we all know, Python’s multi-threading is pseudo-multi-threading because of the existence of GIL. GIL (Global Interpreter Lock) is a mutex lock in the Python interpreter CPython, which ensures that only one thread can execute Python bytecode at any time. In other words, even on a multi-core processor, a Python process can only execute one thread at a time.

The design of the GIL dates back to 1992 to solve thread-safety issues with early Python memory management. At that time, multi-core processors were not yet popular, and single-threaded execution was the mainstream. GIL greatly simplifies Python's memory management, especially the implementation of the reference counting mechanism. There is no need for a complex lock mechanism to protect each object. A global lock solves the thread safety problem. Makes writing C extensions easier without having to consider complex thread synchronization issues.

Advantages of this design:

1. Simple and reliable implementation: Single-threaded execution ensures the security of memory management, reduces the possibility of concurrency bugs such as deadlocks, and simplifies the development of C extensions.

2. Less impact on I/O-intensive applications: **Python releases the GIL when performing I/O operations, so multi-threading can still provide performance improvements** for network requests, file reading and writing, etc. Multi-threaded I/O is a very real requirement.

3. Better single-thread performance: There is no thread switching overhead, no complex lock mechanism is required, and memory management is more efficient.


Disadvantages:

1. Inability to fully utilize multi-core CPU: only one thread can be executed at the same time.

2. Limited performance in computationally intensive tasks: Even with multiple CPU cores, true parallel computing cannot be achieved, so **Python needs to use multiple processes** to handle computationally intensive tasks, such as the following example:
```python
# Computationally intensive tasks may be slower under multi-threading than single-threading
def compute_intensive():
    for i in range(10000000):
        x = i * i
        
#Multi-threaded version may be slower than single-threaded
threads = [Thread(target=compute_intensive) for _ in range(4)]

```
Solution:

1. Use multi-processing instead of multi-threading for computationally intensive tasks:

```python
from multiprocessing import Process # or use mp.spawn

# Use multiple processes to bypass GIL restrictions
processes = [Process(target=compute_intensive) for _ in range(4)]
```

2. Use other Python implementations, or higher versions [Python 3.12](https://www.reddit.com/r/Python/comments/1bcggx9/disabling_the_gil_option_has_been_merged_into/).
  
3. Implement computationally intensive tasks in C/C++: By using extension modules, the GIL can be released in C code.

It should be noted that although the GIL has these limitations, it does not mean that Python is not suitable for developing large applications. In practical applications:

1. Most applications are I/O-bound rather than CPU-bound, and the impact of the GIL is limited.

2. GIL limitations can be circumvented through appropriate architectural design: use a multi-process architecture, use asynchronous programming, hand over computing-intensive tasks to specialized services, or call C/C++ interfaces.


## Ring / Tree Algorithm
- Before starting Collective Communication, NCCL will benchmark different algorithms based on the network communication topology and select the one with the lowest delay. Ring and Tree are the two most common topology algorithms in NCCL, which are often used in ![All-Reduce](./complete-allreduce.svg), but are also used by other operators (Ring All-Gather, All-to-All).
- More complex algorithms include [SHARP](https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf), a multicast extension algorithm (in-network reduction) that uses network swtiches (e.g. [NVSwitch](https://github.com/NVIDIA/nccl/issues/807)) The processor on the CPU performs reduction to avoid offloading data to the CPU/GPU, reducing latency and SM usage.
- In the following analysis, n is the number of GPUs participating in the calculation. NVIDIA usually has 8 GPUs per node, densely connected by high-bandwidth NVLINK + NVSwitch into a complete graph network topology

The vector illustrations drawn by Claude are truly unparalleled...

This picture can understand the Ring/Tree algorithm at a glance.

### Ring Algorithm

**Advantages:**

- High bandwidth utilization: each node receives and sends data at the same time, making full use of hardware bandwidth
- Load balancing: Each node processes the same amount of data, and the network load is evenly distributed
- Simple to implement: fault tolerance and synchronization mechanisms are relatively intuitive**Disadvantages:**

- Latency is linearly related to the number of nodes: it takes 2(n-1) steps to complete an AllReduce
- Not suitable for large-scale clusters: At the scale of thousands of nodes, linearly increasing latency will significantly affect performance, and a few slow straggler nodes can easily become communication bottlenecks.
- Inefficient transmission of small data: relatively large startup overhead

**Expansion/Application:**
- NCCL is more likely to choose the Ring algorithm when there is a single node/a small number of nodes, [and will not mix Ring and Tree](https://github.com/NVIDIA/nccl/issues/471) (Maybe it is laziness/for simple implementation and benchmarking:( .
- Double/2D Ring Topology can be used to efficiently utilize intra-node bandwidth and mask/mitigate inter-node communication delays. NCCL does not use 2D ring, but the paper [LoongTrain](https://arxiv.org/abs/2406.18485) uses 2D ring to accelerate Ring Attention.

### Tree Algorithm

The latency of traditional Tree Algorithm has a logarithmic relationship with the number of nodes. Refer to the picture above to see it clearly.

**Advantages**

- Low latency: logarithmic to the number of nodes: only O(log n) steps are needed to complete communication
- Suitable for large-scale cluster/inter-node communication: excellent performance in large-scale scenarios (such as 24000+ GPU)

**Disadvantages**

- Complex implementation: two complementary binary tree structures need to be maintained
- The advantages of small-scale scenarios are not obvious: when the number of nodes is small, the additional tree structure maintenance overhead may not be worth the gain.
- High requirements on network topology: good network interconnection is required to support tree communication
- Bandwidth utilization is not as good as Ring

**Expansion/Application**
- Sequential parallel algorithm [Tree Attention](https://arxiv.org/abs/2408.04093) uses Tree All-Reduce to accelerate long-context attention calculation during inference. It is more scalable than Ring Attention, but is not suitable for training due to difficulty in overlap calculation and communication.

### Double Binary Tree Algorithm

Starting from NCCL version 2.4, the [Double Binary Tree](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) algorithm is used for cross-node communication with a large number of nodes. Compared with the traditional Tree Algorithm, two complementary binary trees are constructed to balance communication overhead.
![Double Binary Tree](./DBTree.jpg)
1. Complementary structure: Each node is an internal node in one tree (participating in data sending and calculation), and a leaf node in another tree (only participating in data receiving), ensuring that the workload of each node is roughly the same.2. Data segmentation: Divide the data to be transmitted into two parts. Each tree is responsible for processing half of the data, and the two trees work in parallel.

**Advantages**

- Solve bandwidth bottlenecks: avoid single-point bottlenecks through dual-tree structure
- Load balancing: Each node alternates roles in the two trees to ensure load balancing
- Latency advantage: Maintaining O(log n) number of communication steps
- High scalability: suitable for large-scale clusters (24000+ GPUs)
- Good fault tolerance: the impact of a single node failure is small
- High bandwidth utilization: make full use of network bandwidth through data offloading

**Disadvantages**

- Complex implementation: two complementary binary trees need to be maintained
- Additional overhead: Structure maintenance and synchronization overhead is large
- Disadvantages of small scale: when the number of nodes is small, the cost may outweigh the benefits.
- Network sensitive: high requirements on network quality and topology
- Difficulty in debugging: The double-tree structure increases the complexity of debugging

### Usage suggestions

1. **Small scale cluster (< 32 GPU)**
   - Recommended: Traditional Tree Algorithm
   - Reason: Simple implementation, low overhead, and sufficient performance

2. **Medium Scale (32-512 GPU)**
   - Need to choose according to specific scenarios:
   - Focus on simplicity and stability: traditional Tree
   - Performance-focused extension: Double Binary Tree


3. **Large-scale clusters (> 512 GPU)**
   - Recommended: Double Binary Tree
   - Reason: better scalability and load balancing


### Performance comparison

Take NVIDIA's test data as an example, in a cluster of 24756 GPUs:

- Ring Algorithm: delay about 180ms
- Tree Algorithm: Delay about 1ms
- Performance gap close to 180x

## Expand information
- NCCL topology benchmark and selection:
    - https://zhuanlan.zhihu.com/p/718639633
    - Set environment variables to view the topology benchmark results during NCCL init (output a table: latency/bandwidth): `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING`
    - Set environment variables to control the NCCL topology algorithm: `NCCL_ALGO=TREE` or `NCCL_ALGO=RING`
- All-gather algorithm topology: https://github.com/NVIDIA/nccl/issues/1123
- Manually run NCCL performance benchmark: https://github.com/NVIDIA/nccl/issues/569
- Performance advantages of SHARP algorithm:
    - https://www.hpcuserforum.com/presentations/swiss/MellanoxHPCTechnology.pdf
    - https://www.youtube.com/watch?v=is7aBZ1_Op0
- In a multi-machine environment, you can use `ibstatus` to check the Infiniband network card status
- Setting `NCCL_MAX_NCHANNELS=1` can limit the number of channels from the CPU issue kernel to the GPU to 1 (the GPU side scheduler launches the kernel and then executes it in parallel according to different CUDA streams) to ensure that the order of kernel launch is consistent with the CPU side scheduling to avoid the communication kernel starting first, preempting the calculation of kernel SM and then delaying its operation, so that it cannot overlap.    - https://forums.developer.nvidia.com/t/how-many-streams-maximum-number-of-streams/6571/6
    - https://zhuanlan.zhihu.com/p/706805407
