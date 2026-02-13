"""
Compares different decoding strategies across multiple dimensions:
- Speed (tokens/second)
- Quality (perplexity, diversity)
- Output characteristics

This isn't a formal evaluation - just a quick way to eyeball
how different strategies behave on the same prompt.

"""

import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sampler import generate


def calc_perplexity(model, text, tokenizer):
    """
    Lower perplexity = more 'likely' text according to the model.
    Warning: doesn't necessarily mean better quality
    """
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    return torch.exp(outputs.loss).item()


def calc_distinct_n(text, n=2):
    """
    Measure diversity via unique n-grams.
    Higher = more varied, but not always better (could be incoherent).
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    unique_count = len(set(ngrams))
    total_count = len(ngrams)

    return unique_count / total_count if total_count > 0 else 0.0


def run_benchmark():
    """Main benchmark loop"""

    # Setup
    model_name = "gpt2"
    device = "cpu"  # Change to "cuda" if you have a GPU

    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Add context_length if needed
    if not hasattr(model, 'args'):
        model.args = type('Args', (), {'context_length': model.config.n_ctx})()

    # Test prompt
    prompt_text = "The future of AI is"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    eos_id = tokenizer.eos_token_id

    # Strategies to test
    # Format: (display_name, kwargs_dict)
    test_cases = [
        ("Greedy (Baseline)",
         {'strategy': 'greedy'}),

        ("Beam Search (W=3)",
         {'strategy': 'beam_search', 'beam_width': 3}),

        ("Diverse Beam (2 groups)",
         {'strategy': 'diverse_beam_search', 'num_groups': 2, 'beam_width': 4}),

        ("Top-K (K=50)",
         {'strategy': 'top_k', 'top_k': 50}),

        ("Top-P (P=0.9)",
         {'strategy': 'top_p', 'top_p': 0.9}),

        ("Min-P (P=0.05)",
         {'strategy': 'min_p', 'min_p': 0.05}),

        ("Mirostat (Tau=3)",
         {'strategy': 'mirostat', 'tau': 3.0}),

        ("Contrastive (Alpha=0.6)",
         {'strategy': 'contrastive', 'penalty_alpha': 0.6})
    ]

    # Header
    print("\n" + "=" * 105)
    print(f"{'Strategy':<28} | {'Time':<6} | {'PPL':<8} | {'Dist-2':<6} | {'Generated Text'}")
    print("=" * 105)

    # Run each strategy
    for name, kwargs in test_cases:
        try:
            # Time the generation
            start = time.time()
            output = generate(
                model,
                prompt_ids.clone(),
                max_new_tokens=40,
                eos_id=eos_id,
                **kwargs
            )
            duration = time.time() - start

            # Decode and calculate metrics
            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_part = full_text[len(prompt_text):]

            ppl = calc_perplexity(model, full_text, tokenizer)
            diversity = calc_distinct_n(generated_part, n=2)

            # Clean up for display
            snippet = generated_part.replace('\n', ' ')[:50]
            if len(generated_part) > 50:
                snippet += "..."

            print(f"{name:<28} | {duration:>4.2f}s  | {ppl:>6.2f}   | {diversity:>4.2f}   | {snippet}")

        except Exception as e:
            print(f"{name:<28} | FAILED | N/A      | N/A    | Error: {str(e)[:40]}")



if __name__ == "__main__":
    run_benchmark()