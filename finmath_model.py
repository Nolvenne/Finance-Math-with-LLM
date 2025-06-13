#Import libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#Load base model 
model_name = "tiiuae/falcon-rw-1b"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Prevent padding error

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="./offload",  # Specify folder for disk offload
    torch_dtype=torch.float16
)

#Define financial math prompt template 
def build_prompt(question):
    return f"""You are a helpful and accurate AI tutor for financial math problems.

INSTRUCTIONS:
- Solve the question step by step.
- Clearly show any formulas used.
- Finish with 'Final Answer: $___' on a new line.

QUESTION:
{question}

SOLUTION:"""


#Define inference pipeline
def answer_question(question):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()} 
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)