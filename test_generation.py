#!/usr/bin/env python3
"""
Test text generation with the trained model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("Loading trained model...")
    model_path = "./demo_output"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.eval()

    # Test prompts
    prompts = [
        "This is a test",
        "The weather today",
        "In the future",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()