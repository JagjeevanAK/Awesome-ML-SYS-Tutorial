# SGLang multi-modal request life cycle: architecture-level in-depth analysis using Qwen2.5-VL as an example

This document uses `Qwen2.5-VL` as the reference model to provide the ultimate detailed analysis of the multi-modal request processing process within the SGLang framework, going deep into the key functions, data structure conversion and concurrency model levels, aiming to provide developers with a clear understanding at the whiteboard level.

## Core flow chart

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./mm_req_lifecycle.svg" alt="SGLang multi-modal request lifecycle flow chart" style="width: 100%; height: auto;">
</div>

## 1. Service and adaptation layer (`serving_chat.py`)

- **Function**: System entry, converting external OpenAI API format requests into SGLang internal data structures.
- **Input**: Raw HTTP POST request.
- **Output**: `GenerateReqInput` object.
- **Process**:
    - `OpenAIServingChat` receives the request and calls `_process_messages` to apply the chat template.
    - Text and media placeholders (e.g. `<|vision_start|>...<|vision_end|>`) are formatted uniformly.
    - The original media data (such as URL or Base64 encoding) is saved intact in the `GenerateReqInput.image_data` field.

## 2. Tokenizer and multi-modal processor (`tokenizer_manager.py`, `qwen_vl.py`)

- **Role**: The core stage of data preparation and model adaptation.
- **Input**: `GenerateReqInput` object.
- **Output**: Dictionary containing `input_ids`, `mm_items`, `mrope_positions`.
- **Key Process**:
    1. **Concurrent data loading and preprocessing**
       Load image data concurrently, and call Qwen-VL's unique `smart_resize` to scale the image to meet the model input size requirements.
    2. **Tokenization and instant expansion**
       The processor directly replaces the image placeholders in the text with a complete special Token sequence (such as `<|vision_start|>...<|image_pad|>...<|vision_end|>`), that is, the Token expansion has been completed in the Tokenizer stage.
    3. **Calculate M-RoPE location code**
       After generating the expanded `input_ids`, call `MRotaryEmbedding.get_rope_index` to calculate accurate `mrope_positions` based on the input Token and image grid size, providing a basis for subsequent fusion of text and visual features.
    4. **Final Assembly**
       Pack the expanded `input_ids`, the `MultimodalDataItem` list containing `pixel_values`, and `mrope_positions` together and send them to the scheduler.## 3. Scheduler (`scheduler.py`)

- **Function**: Efficient request batch processing and cache management.
- **Input**: Dictionary containing `input_ids`, `mm_items`, `mrope_positions`.
- **Output**: `ScheduleBatch` object.
- **Process**:
    1. Create `Req` and `MultimodalInputs` objects for each request and track the status.
    2. **Radix Cache cache optimization**
       Call `pad_input_ids` to replace special Tokens such as `<|image_pad|>` in `input_ids` with hash values corresponding to `pixel_values`. This hash value serves as a cache key identifier to achieve efficient prefix matching and cache hits for the same image request, even if the text content is different.

## 4. Model execution and feature injection (`model_runner.py`, `qwen2_5_vl.py`)

- **Input**: `ForwardBatch` object.
- **Output**: `logits`.
- **Process**:
    1. `ForwardBatch` created by `ModelRunner`, including `input_ids` and `mrope_positions`.
    2. Call `model.forward()` and pass in `mrope_positions` as a key parameter.
    3. **Dual path feature embedding (M-RoPE enhanced)**
        - **Text path**: `general_mm_embed_routine` gets the general word embedding of the entire `input_ids` (including special Tokens such as `<|vision_start|>`), and uses `mrope_positions` to apply RoPE (rotated position encoding) to ensure that the text and visual parts get precise position information.
        - **Media Path**: Identify the `<|image_pad|>` area and call `get_image_feature`. `Qwen2.5_VisionTransformer` converts `pixel_values` into high-dimensional visual features, and `VisionPatchMerger` is aligned to the language model embedding dimension.
    4. **Accurate injection**: The visual feature embedding covers the word embedding position corresponding to the `<|image_pad|>` Token, and constructs a complete input sequence that combines text and visual information.

## 5. Inference generation and output

- The fused input sequence is sent to the model Transformer layer, and the subsequent process is the same as the plain text model: generating `logits`, sampling and outputting Token, and finally decoding it into text and returning it to the user.## Appendix: Flowchart Mermaid source code```mermaid
graph TD
    A["User request POST /v1/chat/completions Body: messages text, image_url"] --> B1

    subgraph S1 ["1. Service and Adaptation Layer - OpenAI Serving Layer"]
        B1["Receive FastAPI request"]
        B2["Call _process_messages"]
        B3["Apply chat template to generate Prompt"]
        B4["Build GenerateReqInputtext, image_data"]
        B1 --> B2 --> B3 --> B4
    end

    B4 --> C1

    subgraph S2 ["2. Tokenizer and multi-modal processor - Qwen2.5-VL"]
        C1["Receive GenerateReqInput"]
        C2["Call Qwen2_5VLImageProcessor"]
        C3["Concurrent data loading and preprocessing-Concurrent loading of Image/Video- Qwen-VL unique smart_resize"]
        C4["Tokenization and Token expansion replace image placeholders to generate expanded input_ids"]
        C5["Calculate M-RoPE position encoding and call MRotaryEmbedding to generate mrope_positions"]
        C6["Build request input_ids, mm_items, mrope_positions"]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6
    end

    C6 --> D1

    subgraph S3 ["3. Scheduling and caching optimization - scheduler.py"]
        D1["Receive previous stage request"]
        D2["Create Req and MultimodalInputs objects"]
        D3["Radix Cache cache optimization 1. Calculate hash for pixel_values tensor 2. Use hash value as unique identifier 3. Implement efficient prefix matching"]
        D4["Add Req to ScheduleBatch"]
        D1 --> D2 --> D3 --> D4
    end

    D4 --> F1

    subgraph S4 ["4. Model execution preparation - model_runner.py"]
        F1["Receive ScheduleBatch"]
        F2["Create ForwardBatch to prepare low-order GPU tensors containing mrope_positions"]
        F3["Call model.forward"]
        F1 --> F2 --> F3
    end

    F3 --> G1

    subgraph S5 ["5. Model forward propagation - Qwen2.5-VL"]
        G1["Call general_mm_embed_routine"]
        G2["Get text Token Embeddings using mrope_positions for position encoding"]
        G3["Identify image placeholder area"]
``````mermaid
        G4["Call get_image_feature input: MultimodalDataItem"]
        G5["Qwen2.5 Vision Transformerpixel_values -> High-dimensional visual features"]
        G6["Vision Patch Merger high-dimensional features -> language model dimensions"]
        G7["Inject Embedding to cover the projected visual Embedding to the placeholder area"]
        G1 --> G2 --> G3 --> G4 --> G5 --> G6 --> G7
    end

    G7 --> H["Merged Embeddings"]
    H --> I["LLM Backbone"]
    I --> J["Logits"]
    J --> K["Sampling -> Output Token"]
```