# Knowledge Elicitation Prompting Approach (KEPA)

The agents use large language models (LLMs) to analyze reports and predict T and N stages of breast cancer based on learned rules (memory).

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
CUDA_VISIBLE_DEVICES="2,3,4,5" python -m vllm.entrypoints.openai.api_server \
--model mistralai/Mixtral-8x7B-Instruct-v0.1 \
--download-dir /path/to/cache/ \
--tensor-parallel-size 4 \
--disable-custom-all-reduce \
--enforce-eager
```

- `CUDA_VISIBLE_DEVICES`: Specifies the CUDA devices to use.
- `--model`: Specifies the model to use.
- `--download-dir`: Sets the directory to the local model.

## Usage

### ChoiceAgent

`ChoiceAgent` is designed for zero-shot prompting, where the model simply selects an answer from given choices (T1 ~ T4, N0 ~ N3), without memory.

#### Initialization

```python
from openai import OpenAI
from prompt import zs_predict_prompt_n03
from agent import ChoiceAgent

client = OpenAI(api_key='', base_url = "http://localhost:8000/v1")

agent = ChoiceAgent(
    client=client,
    model='your_model_name',
    prompt_template= zs_predict_prompt_n03,
    choices= ['N0', 'N1', 'N2','N3'],
    label='n'
)

```

#### Running the Agent

```python
dataset = pd.read_csv('path_to_your_dataset.csv')
result = agent.run(dataset, temperature=0.1)
```

### ConditionalMemoryAgent

`ConditionalMemoryAgent` builds and utilizes memory from training data to assist in interpreting test data.

#### Initialization

```python
from openai import OpenAI

client = OpenAI(api_key='your_api_key')

agent = ConditionalMemoryAgent(
    client=client,
    model='your_model_name',
    prompt_template_dict={
        'initialized_prompt': 'Your initial prompt template',
        'learning_prompt': 'Your learning prompt template',
        'testing_prompt': 'Your testing prompt template'
    },
    schema_dict={
        'learning_schema': 'Your learning schema',
        'testing_schema': 'Your testing schema'
    },
    label='YourLabel'
)
```

#### Training

```python
training_data = pd.read_csv('path_to_your_training_data.csv')
trained_data = agent.train(training_data, temperature=0.1, threshold=80)
```

#### Testing

Testing involves using the agent object's internal memory, which has been built during the training phase, to interpret and predict cancer stages from new reports.

Example:

```python
testing_data = pd.read_csv('path_to_your_testing_data.csv')
test_result = agent.test(testing_data)
```

#### Dynamic Testing

Dynamic testing uses external memory tuples, which can be created from a separate training phase. 

Example:

```python
memory_tup = [...]  # List of (index, memory) tuples
test_result = agent.dynamic_test(testing_data, memory_tup)
```

