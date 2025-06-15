# import libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from finance_formulas import (
    simple_interest,
    compound_interest,
    parse_simple_interest_input,
    parse_compound_interest_input
)

# Load LLM model
model_name = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="./offload",
    torch_dtype=torch.float16
)

# Prompt template for fallback
def build_prompt(question):
    return f"""You are a helpful and accurate AI tutor for financial math problems.

INSTRUCTIONS:
- Solve the question step by step.
- Clearly show any formulas used.
- Finish with 'Final Answer: $___' on a new line.

QUESTION:
{question}

SOLUTION:"""

# prompt for LLM
def get_llm_explanation(P, R, T, result, formula_type="simple_interest"):
    if formula_type == "simple_interest":
        prompt = (
            f"Solve the following financial math problem step-by-step:\n"
            f"What is the simple interest on ${P} at {R}% for {T} years?\n"
            f"The correct answer is ${result:.2f}."
        )
    elif formula_type == "compound_interest":
        prompt = (
            f"Solve the following financial math problem step-by-step:\n"
            f"What is the compound interest on ${P} at {R}% for {T} years?\n"
            f"The correct answer is ${result:.2f}."
        )
    else:
        return None

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        explanation = tokenizer.decode(output[0], skip_special_tokens=True)

        # Check for prompt echo or failure to generate
        if "Final Answer" not in explanation and explanation.strip().startswith("Solve"):
            raise ValueError("LLM did not generate a real explanation.")
        return explanation

    except Exception as e:
        # Fall back to Python based explanation
        if formula_type == "simple_interest":
            return (
                f"To calculate simple interest, use the formula:\n"
                f"SI = P × R × T / 100\n"
                f"Where:\n"
                f"- P = ${P}, R = {R}%, T = {T} years\n"
                f"SI = {P} × {R} × {T} / 100 = ${result:.2f}\n"
                f"Final Answer: ${result:.2f}"
            )
        elif formula_type == "compound_interest":
            return (
                f"To calculate compound interest, use the formula:\n"
                f"CI = P × (1 + R/100)^T - P\n"
                f"Where:\n"
                f"- P = ${P}, R = {R}%, T = {T} years\n"
                f"CI = {P} × (1 + {R/100})^{T} - {P} = ${result:.2f}\n"
                f"Final Answer: ${result:.2f}"
            )
        else:
            return "Sorry, I couldn't explain this problem."

# Main inference function
def answer_question(question):
    # Simple Interest
    P, R, T = parse_simple_interest_input(question)
    if P and R and T:
        si = simple_interest(P, R, T)
        return get_llm_explanation(P, R, T, si, formula_type="simple_interest")

    # Compound Interest
    P, R, T = parse_compound_interest_input(question)
    if P and R and T:
        ci = compound_interest(P, R, T)
        return get_llm_explanation(P, R, T, ci, formula_type="compound_interest")

    # Fallback to full LLM if nothing matches
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
