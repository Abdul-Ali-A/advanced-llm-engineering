# <----------------------------[Using TGI(Text Generation Infrence)]---------------------------->
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080/")

# Only Generating Text:
response = client.text_generation(
    "Tell top 5 rare fruits",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
    stop_sequences=[],
)
print(response.generated_text)

# Chat-Style Format:
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell top 5 rare fruits"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# <----------------------------[Using vLLM]---------------------------->
from vllm import LLM, SamplingParams

# Initializing Model:
llm = LLM(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    gpu_memory_utilization=0.85,
    max_num_batched_tokens=8192,
    max_num_seqs=256,
    block_size=16,
)

# Configure sampling parameters:
sampling_params = SamplingParams(
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    max_tokens=100,  # Maximum length
    presence_penalty=1.1,  # Reduce repetition
    frequency_penalty=1.1,  # Reduce repetition
    stop=["\n\n", "###"],  # Stop sequences
)

# Only Generating Text:
prompt = "Write a creative story"
outputs = llm.generate(prompt, sampling_params)
print(outputs[0].outputs[0].text)

# Chat-Style Format:
chat_prompt = [
    {"role": "system", "content": "You are a creative storyteller."},
    {"role": "user", "content": "Write a creative story"},
]

formatted_prompt = llm.get_chat_template()(chat_prompt)  # Uses model's chat template
outputs = llm.generate(formatted_prompt, sampling_params)
print(outputs[0].outputs[0].text)
