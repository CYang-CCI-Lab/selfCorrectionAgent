# Knowledge Elicitation Prompting Approach (KEPA)

## Overview

This repository contains the implementation of the Knowledge Elicitation Prompting Approach (KEPA), a method designed to elicit and utilize domain-specific knowledge from pathology reports to identify cancer stages. KEPA leverages a large language model (LLM) to generate relevant reasoning, predict cancer stages, and update long-term memory with learned rules.


## Server Setup

To run the model, we host a vllm server using the `Mixtral-8x7B-Instruct-v0.1` model. The server listens on port 8000.

- recommend to create a new conda environment
- pip install the packages:
    
```bash
# (Recommended) Create a new conda environment.
conda create -n myenv python=3.9 -y
conda activate myenv
```

```bash
# Install vLLM with CUDA 12.1.
pip install vllm
```

```bash
# Install flash-attention to improve inference speed
pip install flash-attn
```

```bash
CUDA_VISIBLE_DEVICES="0,1,3,4" python -m vllm.entrypoints.openai.api_server \
--model mistralai/Mixtral-8x7B-Instruct-v0.1 \
--download-dir /path/to/cache/ \
--tensor-parallel-size 4 \
--disable-custom-all-reduce \
--enforce-eager
```

- `CUDA_VISIBLE_DEVICES`: Specifies the CUDA devices to use.
- `--model`: Specifies the model to use.
- `--download-dir`: Sets the directory to the local model.

## Key Files

- **src/agent.py**: Contains the implementation of various agents used in the KEPA approach, including `ZSAgent`, `ZSCOTAgent`, `KEPATrainAgent`, and `KEPATestAgent`.

- **src/prompt.py**: Defines prompt templates used for initial elicitation, subsequent updates, and testing of cancer stage predictions.





