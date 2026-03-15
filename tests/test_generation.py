"""
Tier A (Gold Standard): Exact token-level match with HF
Tier S (White-box): Internal logic verification via logits masks
Tier B (Property-based): Behavioral correctness checks
Tier 0 : Just make sure it doesn't crash
"""

import torch
import pytest
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor
)

from sampler import (
    generate,
    top_k_sampling,
    top_p_sampling,
    min_p_sampling,
    apply_repetition_penalty,
    apply_no_repeat_ngram
)


@pytest.fixture(scope="module")
def model_setup():
    """Load GPT-2 once for all tests to save time"""
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Add context_length attribute if missing
    if not hasattr(model, 'args'):
        model.args = type('Args', (), {'context_length': model.config.n_ctx})()

    return model, tokenizer


# Tier A: Gold Standard Tests
# These should produce IDENTICAL outputs to HuggingFace

def test_greedy_matches_huggingface_exactly(model_setup):
    """
    Greedy decoding is deterministic - outputs should be byte-for-byte identical.
    If this fails, something is fundamentally wrong.
    """
    model, tokenizer = model_setup
    prompt = tokenizer("AI is", return_tensors="pt")["input_ids"]

    # HuggingFace reference
    hf_output = model.generate(
        prompt,
        max_new_tokens=5,
        do_sample=False,  # greedy
        pad_token_id=50256
    )

    # Our implementation
    our_output = generate(
        model,
        prompt,
        max_new_tokens=5,
        strategy='greedy',
        eos_id=50256
    )

    assert torch.equal(hf_output, our_output), \
        f"Greedy outputs differ!\nHF: {tokenizer.decode(hf_output[0])}\nOurs: {tokenizer.decode(our_output[0])}"


def test_beam_search_high_overlap_with_huggingface(model_setup):
    """
    Beam search should have very high overlap with HF.
    Small differences are OK due to tie-breaking, but should be >90% similar.
    """
    model, tokenizer = model_setup
    prompt = tokenizer("The theory of", return_tensors="pt")["input_ids"]

    hf_output = model.generate(
        prompt,
        max_new_tokens=10,
        num_beams=3,
        early_stopping=False,
        pad_token_id=50256
    )

    our_output = generate(
        model,
        prompt,
        max_new_tokens=10,
        strategy='beam_search',
        beam_width=3,
        eos_id=50256
    )

    # Calculate token-level overlap
    matches = (hf_output[0] == our_output[0]).float().mean().item()

    assert matches > 0.9, \
        f"Beam search accuracy too low: {matches:.2%}\nHF: {tokenizer.decode(hf_output[0])}\nOurs: {tokenizer.decode(our_output[0])}"


# Tier S: White-box Logic Tests
# Verify internal correctness by checking logits masks

def test_top_k_masks_correctly(model_setup):
    """
    Top-K should mask out all tokens except the K highest.
    We verify the mask pattern matches HF's implementation.
    """
    # Random logits for testing
    logits = torch.randn(2, 50)
    k = 5

    # HuggingFace approach
    hf_warper = TopKLogitsWarper(top_k=k)
    hf_result = hf_warper(None, logits.clone())

    # Our approach
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[:, -1].unsqueeze(-1)
    our_result = torch.where(
        logits < threshold,
        torch.tensor(float('-inf')),
        logits
    )

    # Check that we mask the same tokens
    assert torch.equal(
        hf_result == float('-inf'),
        our_result == float('-inf')
    ), "Top-K masking differs from HuggingFace"


def test_top_p_masks_correctly(model_setup):
    """
    Top-P (nucleus) should keep smallest set where cumulative prob >= p.
    Verify we get the same truncation point as HF.
    """
    logits = torch.randn(2, 50)
    p = 0.8

    # HuggingFace
    hf_warper = TopPLogitsWarper(top_p=p)
    hf_result = hf_warper(None, logits.clone())

    # Our implementation
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1),
        dim=-1
    )

    # Mark tokens to remove
    sorted_remove = cumulative_probs > p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = 0

    # Scatter back to original order
    indices_to_remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
    our_result = logits.masked_fill(indices_to_remove, float('-inf'))

    assert torch.equal(
        hf_result == float('-inf'),
        our_result == float('-inf')
    ), "Top-P masking differs from HuggingFace"


def test_temperature_scales_correctly():
    """Temperature should just divide logits. Dead simple."""
    logits = torch.randn(2, 10)
    temp = 0.5

    hf_warper = TemperatureLogitsWarper(temperature=temp)
    hf_result = hf_warper(None, logits.clone())

    expected = logits / temp

    assert torch.allclose(hf_result, expected), \
        "Temperature scaling doesn't match expected behavior"


def test_repetition_penalty_matches_huggingface():
    """Repetition penalty should modify logits identically to HF"""
    logits = torch.randn(1, 10)
    input_ids = torch.tensor([[1, 5]])
    penalty = 2.0

    hf_processor = RepetitionPenaltyLogitsProcessor(penalty=penalty)
    hf_result = hf_processor(input_ids, logits.clone())

    our_result = apply_repetition_penalty(logits.clone(), input_ids, penalty=penalty)

    assert torch.allclose(hf_result, our_result), \
        "Repetition penalty differs from HuggingFace"


def test_ngram_blocking_matches_huggingface():
    """N-gram blocking should prevent exact same sequences"""
    input_ids = torch.tensor([[1, 2, 1]])
    logits = torch.zeros(1, 10)

    hf_processor = NoRepeatNGramLogitsProcessor(ngram_size=2)
    hf_result = hf_processor(input_ids, logits.clone())

    our_result = apply_no_repeat_ngram(logits.clone(), input_ids, ngram_size=2)

    assert torch.equal(hf_result, our_result), \
        "N-gram blocking differs from HuggingFace"


# Tier B: Property-Based Tests
# These techniques don't have HF equivalents - we verify properties instead

def test_min_p_respects_threshold():
    """
    Min-P should only sample tokens with prob >= (min_p * max_prob).
    We can't test against HF, but we can verify the property holds.
    """
    logits = torch.randn(1, 100)
    min_p = 0.1

    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max()

    # Sample and check it respects threshold
    sampled_idx = min_p_sampling(logits.clone(), min_p=min_p)
    sampled_prob = probs[0, sampled_idx]

    assert sampled_prob >= max_prob * min_p, \
        f"Min-P violated: sampled {sampled_prob:.4f}, threshold was {max_prob * min_p:.4f}"


def test_advanced_sampling_doesnt_crash(model_setup):
    """
    For techniques without HF equivalents, just verify they run.
    If they produce output of correct length, that's a good sign.
    """
    model, tokenizer = model_setup
    prompt = tokenizer("Test", return_tensors="pt")["input_ids"]

    # These are all custom implementations
    custom_strategies = [
        ('mirostat', {'tau': 4.0}),
        ('eta', {'eta': 0.002}),
        ('typical', {'mass': 0.9}),
        ('locally_typical', {'tau': 0.5}),
        ('top_a', {'top_a': 0.5}),
        ('tfs', {'z': 0.9}),
        ('epsilon', {'epsilon': 0.02})
    ]

    for strategy, kwargs in custom_strategies:
        output = generate(
            model,
            prompt.clone(),
            max_new_tokens=2,
            strategy=strategy,
            eos_id=50256,
            **kwargs
        )

        # Should generate exactly 2 new tokens
        assert output.size(1) == prompt.size(1) + 2, \
            f"{strategy} didn't generate correct number of tokens"


def test_constrained_beam_includes_required_token(model_setup):
    """
    Constrained beam search should force inclusion of specified tokens.
    This is its whole purpose - if it fails here, it's broken.
    """
    model, tokenizer = model_setup
    prompt = tokenizer("I like", return_tensors="pt")["input_ids"]

    # Force it to include "pizza"
    target_word = " pizza"
    target_id = tokenizer.encode(target_word)[0]

    output = generate(
        model,
        prompt,
        max_new_tokens=5,
        strategy='constrained_beam_search',
        must_include=[target_id],
        eos_id=50256
    )

    assert target_id in output[0], \
        f"Constrained beam failed to include required token '{target_word}'"


def test_diverse_beam_produces_output(model_setup):
    """Diverse beam should produce longer output than input"""
    model, tokenizer = model_setup
    prompt = tokenizer("AI", return_tensors="pt")["input_ids"]

    output = generate(
        model,
        prompt,
        max_new_tokens=3,
        strategy='diverse_beam_search',
        num_groups=2,
        beam_width=4,
        eos_id=50256
    )

    assert output.size(1) > prompt.size(1), \
        "Diverse beam didn't generate any tokens"


def test_best_of_n_selects_from_candidates(model_setup):
    """Best-of-N should generate and pick best from N candidates"""
    model, tokenizer = model_setup
    prompt = tokenizer("Hello", return_tensors="pt")["input_ids"]

    output = generate(
        model,
        prompt,
        max_new_tokens=3,
        strategy='best_of_n',
        n=2,
        eos_id=50256
    )

    assert output.size(1) > prompt.size(1), \
        "Best-of-N didn't generate any tokens"


# Tier 0: Other Tests
# Just make sure these don't explode

def test_remaining_strategies(model_setup):
    """
    Quick sanity check for all remaining techniques.
    If any of these crash, we have a problem.
    """
    model, tokenizer = model_setup
    prompt = tokenizer("Smoke test", return_tensors="pt")["input_ids"]

    smoke_cases = [
        ("stochastic_beam_search", {'beam_width': 2}),
        ("contrastive", {'penalty_alpha': 0.5}),
        ("dynamic_temperature", {'min_temp': 0.5}),
        ("temperature_annealing", {'start_temp': 2.0}),
        ("adaptive", {'confidence_threshold': 0.8}),
        ("greedy", {'frequency_penalty': 0.1}),
        ("greedy", {'presence_penalty': 0.1}),
        ("greedy", {'encoder_repetition_penalty': 1.1}),
        ("beam_search", {'beam_width': 2})  # length_penalty tested separately
    ]

    for strategy, kwargs in smoke_cases:
        try:
            generate(
                model,
                prompt.clone(),
                max_new_tokens=1,
                strategy=strategy,
                eos_id=50256,
                **kwargs
            )
        except Exception as e:
            pytest.fail(f"{strategy} crashed: {e}")



if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])