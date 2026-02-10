import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer



# GPT-2 as a test tool


class GPT2Wrapper(torch.nn.Module):
    """
    Simple wrapper around HuggingFace's GPT-2 to match our interface.
    Makes it easier to swap in different models later.
    """
    def __init__(self, model_name='gpt2'):
        super().__init__()
        print(f"Loading {model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Create a simple args object to expose context_length
        self.args = type('Args', (), {
            'context_length': self.model.config.n_ctx
        })()

    def forward(self, idx):
        output = self.model(idx)
        return output.logits


# Main Generation Loop

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    max_new_tokens: int,
    eos_id: int = 7,
    strategy: str = 'greedy',
    **decode_kwargs
) -> torch.Tensor:
    """
    Universal text generation function supporting multiple decoding strategies.

    This is the main entry point. It handles:
    - Strategy dispatch
    - Token-by-token generation loop
    - Penalty applications
    - Early stopping on EOS

    Args:
        model: Language model (must have .forward() returning logits)
        inputs: Initial token IDs [batch_size, seq_len]
        max_new_tokens: How many new tokens to generate
        eos_id: End-of-sequence token ID
        strategy: Which decoding strategy to use
        **decode_kwargs: Strategy-specific parameters

    Returns:
        Generated sequences [batch_size, original_len + generated_len]
    """

    # Map strategy names to their implementation functions
    strategies = {
        'greedy': greedy_decode,
        'temperature': multinomial_sampling,
        'top_k': top_k_sampling,
        'top_p': top_p_sampling,
        'min_p': min_p_sampling,
        'typical': typical_sampling,
        'top_a': top_a_sampling,
        'tfs': tail_free_sampling,
        'eta': eta_sampling,
        'mirostat': mirostat_sampling,
        'epsilon': epsilon_sampling,
        'locally_typical': locally_typical_sampling,
        'top_k_top_p': combined_top_k_top_p,
        'dynamic_temperature': dynamic_temperature,
        'temperature_annealing': temperature_annealing,
        'adaptive': adaptive_sampling,
        'contrastive': contrastive_sampling,
        'length_penalty': length_penalty_sampling,
    }

    # Beam search variants need special handling since they control the loop themselves
    if strategy == 'beam_search':
        return beam_search_decode(model, inputs, max_new_tokens, eos_id=eos_id, **decode_kwargs)
    elif strategy == 'diverse_beam_search':
        return diverse_beam_search(model, inputs, max_new_tokens, eos_id=eos_id, **decode_kwargs)
    elif strategy == 'constrained_beam_search':
        return constrained_beam_search(model, inputs, max_new_tokens, eos_id=eos_id, **decode_kwargs)
    elif strategy == 'stochastic_beam_search':
        return stochastic_beam_search(model, inputs, max_new_tokens, eos_id=eos_id, **decode_kwargs)
    elif strategy == 'best_of_n':
        return best_of_n_sampling(model, inputs, max_new_tokens, eos_id=eos_id, **decode_kwargs)

    if strategy not in strategies:
        raise ValueError(f"Strategy '{strategy}' not implemented. Available: {list(strategies.keys())}")

    sampling_fn = strategies[strategy]

    print(f"--- Starting generation (strategy: {strategy}) ---")
    batch_size = inputs.shape[0]
    finished = torch.zeros(batch_size, 1, dtype=torch.bool, device=inputs.device)

    # Initialize Mirostat state (only used if strategy='mirostat')
    target_tau = decode_kwargs.get('tau', 5.0)
    mu = torch.full((batch_size, 1), 2 * target_tau, device=inputs.device)
    original_input = inputs.clone()

    # Main generation loop
    for step in range(max_new_tokens):
        # Truncate context if needed
        idx_cond = (inputs if inputs.size(1) <= model.args.context_length
                   else inputs[:, -model.args.context_length:])

        logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]

        # Apply frequency and presence penalties
        freq_p = decode_kwargs.get('frequency_penalty', 0.0)
        pres_p = decode_kwargs.get('presence_penalty', 0.0)
        if freq_p != 0.0 or pres_p != 0.0:
            next_token_logits = apply_frequency_presence_penalty(
                next_token_logits, inputs, freq_p, pres_p
            )

        # Apply repetition penalty
        rep_p = decode_kwargs.get('repetition_penalty', 1.0)
        if rep_p != 1.0:
            next_token_logits = apply_repetition_penalty(next_token_logits, inputs, rep_p)

        # Apply encoder repetition penalty (penalizes tokens from original prompt)
        enc_rep_p = decode_kwargs.get('encoder_repetition_penalty', 1.0)
        if enc_rep_p != 1.0:
            next_token_logits = apply_encoder_repetition_penalty(
                next_token_logits, original_input, inputs, enc_rep_p
            )

        # Block repeated n-grams
        ngram_size = decode_kwargs.get('no_repeat_ngram_size', 0)
        if ngram_size > 0:
            next_token_logits = apply_no_repeat_ngram(next_token_logits, inputs, ngram_size)

        # Inject context information needed by some strategies
        decode_kwargs['step'] = step
        decode_kwargs['max_steps'] = max_new_tokens
        decode_kwargs['sequences'] = inputs
        decode_kwargs['current_length'] = inputs.size(1)

        # Sample next token
        if strategy == 'mirostat':
            decode_kwargs.setdefault('tau', target_tau)
            decode_kwargs.setdefault('eta', 0.1)
            next_token, mu = mirostat_sampling(next_token_logits, mu=mu, **decode_kwargs)
        else:
            next_token = sampling_fn(next_token_logits, **decode_kwargs)

        # Ensure correct shape
        if next_token.dim() == 1:
            next_token = next_token.unsqueeze(-1)

        # Replace finished sequences with EOS
        next_token = torch.where(finished, torch.tensor(eos_id, device=inputs.device), next_token)
        inputs = torch.cat((inputs, next_token), dim=1)

        # Update finished status
        is_eos = (next_token == eos_id)
        finished = finished | is_eos

        if finished.all():
            break

    return inputs


# Sampling Strategies

def greedy_decode(logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Greedy Decoding

    [Intuition]:
    Always picks the most likely token. Deterministic and fast, but can be
    repetitive and boring. Good for tasks where you want consistent output
    (e.g., translation, summarization).

    [Formula]:
    .. math::
        x_t = \\arg\\max_{v \\in V} P(v | x_{<t})

    where:
        - :math:`x_t` is the selected token at step t
        - :math:`V` is the vocabulary
        - :math:`P(v | x_{<t})` is the probability of token v given previous context

    [Implementation]:
    Simple argmax over the logits.
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


def multinomial_sampling(logits: torch.Tensor, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Multinomial Sampling with Temperature

    [Intuition]:
    Temperature controls how "sharp" or "flat" the probability distribution is:
    - temperature < 1.0: Makes distribution sharper (conservative, favors likely tokens)
    - temperature > 1.0: Makes distribution flatter (creative, more randomness)
    - temperature = 1.0: Uses raw model probabilities

    [Formula]:
    .. math::
        P_i = \\frac{\\exp(\\text{logits}_i / T)}{\\sum_j \\exp(\\text{logits}_j / T)}

    where T is the temperature parameter.

    [Implementation]:
    1. Scale logits by temperature
    2. Convert to probabilities via softmax
    3. Sample from the distribution

    Args:
        logits: Raw model scores [batch_size, vocab_size]
        temperature: Sampling temperature (must be > 0)
    """
    if temperature < 1e-5:
        return greedy_decode(logits)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_k_sampling(logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Top-K Sampling

    [Intuition]:
    Only consider the K most likely tokens. This cuts off the "long tail" of
    unlikely tokens that might make the text incoherent, while still allowing
    some diversity in generation.

    [Formula]:
    Let :math:`V^{(K)} \\subset V` be the set of K tokens with highest probability.
    The adjusted distribution is:

    .. math::
        P'(x) = \\begin{cases}
        P(x) / \\sum_{y \\in V^{(K)}} P(y) & \\text{if } x \\in V^{(K)} \\\\
        0 & \\text{otherwise}
        \\end{cases}

    [Implementation]:
    1. Find the K-th largest logit value (threshold)
    2. Set all logits below threshold to -infinity
    3. Apply temperature and sample

    Args:
        top_k: Number of top tokens to keep (typically 40-100)
        temperature: Sampling temperature
    """
    top_k_values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = top_k_values[:, -1].unsqueeze(-1)

    logits_masked = logits.masked_fill(logits < threshold, float('-inf'))
    return multinomial_sampling(logits_masked, temperature=temperature)


def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Top-P Sampling (Nucleus Sampling)

    [Intuition]:
    Instead of a fixed K, dynamically choose how many tokens to consider based
    on cumulative probability. Stop when you've covered P% of the probability mass.

    This is adaptive:
    - When model is confident: Few tokens are needed to reach P
    - When model is uncertain: More tokens are included

    Most commonly used in modern LLMs (GPT-3, GPT-4, etc.)

    [Formula]:
    Let :math:`V^{(P)}` be the smallest set such that:

    .. math::
        \\sum_{x \\in V^{(P)}} P(x) \\ge P

    Then renormalize probabilities over this set.

    [Implementation]:
    1. Sort tokens by probability (descending)
    2. Compute cumulative probabilities
    3. Find cutoff where cumsum > top_p
    4. Mask out tokens beyond cutoff

    Args:
        top_p: Cumulative probability threshold (typically 0.9-0.95)
        temperature: Sampling temperature
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Create mask for tokens to remove
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep at least one token (shift mask right)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter mask back to original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove
    )

    logits_masked = logits.masked_fill(indices_to_remove, float('-inf'))
    return multinomial_sampling(logits_masked, temperature=temperature)


def min_p_sampling(logits: torch.Tensor, min_p: float = 0.05, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Min-P Sampling

    [Intuition]:
    A modern alternative to Top-P. Instead of a fixed cumulative threshold,
    set a threshold relative to the most likely token.

    Keep tokens with probability >= (min_p * max_probability)

    This naturally adapts to the model's confidence level.

    [Formula]:
    Let :math:`P_{\\text{max}} = \\max(P(x))`. Keep set:

    .. math::
        V_{\\text{kept}} = \\{ x \\in V \\mid P(x) \\ge \\text{min_p} \\times P_{\\text{max}} \\}

    Args:
        min_p: Relative threshold (typically 0.05-0.1)
        temperature: Sampling temperature
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = torch.max(probs, dim=-1, keepdim=True)
    tokens_threshold = top_probs * min_p

    logits_masked = logits.masked_fill(probs < tokens_threshold, float('-inf'))
    return multinomial_sampling(logits_masked, temperature=temperature)


def typical_sampling(logits: torch.Tensor, mass: float = 0.9, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Typical Sampling

    [Intuition]:
    Based on information theory. Instead of choosing high-probability tokens,
    choose tokens whose "surprisal" (negative log probability) is close to
    the expected surprisal (entropy).

    This avoids both:
    - Too obvious tokens (boring)
    - Too surprising tokens (incoherent)

    [Formula]:
    Entropy:

    .. math::
        H(P) = -\\sum_{x} P(x) \\log P(x)

    Keep tokens where:

    .. math::
        |-\\log P(x) - H(P)| < \\text{threshold}

    [Implementation]:
    1. Compute entropy H = -sum(p * log p)
    2. Compute distance from entropy: |−log p − H|
    3. Keep tokens with smallest distance (most "typical")

    Args:
        mass: Cumulative probability mass to keep (like top_p)
        temperature: Sampling temperature
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Compute entropy (handle log(0) gracefully)
    p_log_p = probs * log_probs
    p_log_p = torch.nan_to_num(p_log_p, nan=0.0)
    entropy = -torch.sum(p_log_p, dim=-1, keepdim=True)

    # Compute distance from entropy
    negative_log_probs = -log_probs
    dist = torch.abs(negative_log_probs - entropy)

    # Sort by distance and apply cumulative mass threshold
    sorted_dist, sorted_indices = torch.sort(dist, dim=-1)
    sorted_probs = torch.gather(probs, dim=-1, index=sorted_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > mass
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove
    )

    logits_masked = logits.masked_fill(indices_to_remove, float('-inf'))
    return multinomial_sampling(logits_masked, temperature=temperature)


def top_a_sampling(logits: torch.Tensor, top_a: float = 0.0, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Top-A Sampling

    [Intuition]:
    Adaptive truncation based on the squared probability of the top token.
    The threshold automatically adjusts to model confidence.

    [Formula]:
    .. math::
        \\text{Threshold} = P_{\\text{max}}^2 \\times \\text{top_a}

    Args:
        top_a: Scaling factor (typically 0.0-1.0)
        temperature: Sampling temperature
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = torch.max(probs, dim=-1, keepdim=True)

    limit_probs = (top_probs ** 2) * top_a

    logits_masked = torch.where(
        probs < limit_probs,
        torch.tensor(float('-inf'), device=logits.device),
        logits
    )

    return multinomial_sampling(logits_masked, temperature=temperature)


def tail_free_sampling(logits: torch.Tensor, z: float = 0.95, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Tail Free Sampling (TFS)

    [Intuition]:
    Instead of looking at absolute probabilities, look at how fast they drop off.
    Find the "elbow" where probability starts dropping rapidly (the long tail begins).

    Uses second derivative to detect this inflection point.

    [Formula]:
    Sort probabilities: :math:`p_1 \\ge p_2 \\ge ... \\ge p_n`

    First derivative:

    .. math::
        d^{(1)}_i = p_i - p_{i+1}

    Second derivative:

    .. math::
        d^{(2)}_i = d^{(1)}_i - d^{(1)}_{i+1}

    Keep tokens until:

    .. math::
        \\sum_{i=1}^{k} \\frac{|d^{(2)}_i|}{\\sum_j |d^{(2)}_j|} > z

    [Implementation]:
    1. Sort probabilities descending
    2. Compute first derivative (differences between consecutive probs)
    3. Compute second derivative (rate of change of first derivative)
    4. Cut off when cumulative absolute second derivative > z

    Args:
        z: Cumulative second derivative threshold (typically 0.90-0.95)
        temperature: Sampling temperature
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # First derivative
    d1 = sorted_probs[:, :-1] - sorted_probs[:, 1:]

    # Second derivative
    d2 = d1[:, :-1] - d1[:, 1:]

    # Normalize absolute second derivative
    d2_abs = torch.abs(d2)
    d2_norm = d2_abs / (torch.sum(d2_abs, dim=-1, keepdim=True) + 1e-8)

    cumulative_d2 = torch.cumsum(d2_norm, dim=-1)

    # Find cutoff for each batch element
    mask = torch.zeros_like(logits, dtype=torch.bool)

    for i in range(logits.size(0)):
        cutoff_idx = (cumulative_d2[i] > z).nonzero()
        if len(cutoff_idx) > 0:
            k = cutoff_idx[0].item() + 2
        else:
            k = logits.size(-1)

        top_indices = sorted_indices[i, :k]
        mask[i, top_indices] = True

    logits_masked = torch.where(
        mask,
        logits,
        torch.tensor(float('-inf'), device=logits.device)
    )

    return multinomial_sampling(logits_masked, temperature=temperature)


def eta_sampling(logits: torch.Tensor, eta: float = 0.002, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Eta Sampling (η-sampling)

    [Intuition]:
    Entropy-based dynamic truncation. The threshold adapts based on how
    uncertain the model is (measured by entropy).

    [Formula]:
    .. math::
        \\epsilon = \\min(\\eta, \\sqrt{\\eta \\cdot H(p)})

    where H(p) is the entropy of the distribution.

    Args:
        eta: Base threshold parameter (typically 0.001-0.01)
        temperature: Sampling temperature
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    p_log_p = torch.nan_to_num(probs * log_probs, nan=0.0)
    entropy = -torch.sum(p_log_p, dim=-1, keepdim=True)

    epsilon = torch.minimum(
        torch.tensor(eta, device=logits.device),
        torch.sqrt(eta * entropy)
    )

    logits_masked = logits.masked_fill(probs < epsilon, float('-inf'))
    return multinomial_sampling(logits_masked, temperature=temperature)


def epsilon_sampling(logits: torch.Tensor, epsilon: float = 0.001, temperature: float = 1.0, **kwargs) -> torch.Tensor:
    """
    [Strategy]: Epsilon Sampling

    [Intuition]:
    Simple fixed threshold. Remove all tokens with probability below epsilon.
    Unlike Min-P, this uses an absolute threshold rather than relative.

    Useful when you know roughly what probability floor makes sense for your task.

    [Formula]:
    .. math::
        V_{\\text{kept}} = \\{ x \\in V \\mid P(x) \\ge \\epsilon \\}

    Args:
        epsilon: Fixed probability threshold (typically 0.001-0.01)
        temperature: Sampling temperature
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    mask = probs >= epsilon

    # Safety: if everything is masked, fall back to greedy
    if not mask.any():
        return greedy_decode(logits)

    logits_masked = logits.masked_fill(~mask, float('-inf'))
    return multinomial_sampling(logits_masked, temperature=temperature)


def locally_typical_sampling(
        logits: torch.Tensor,
        tau: float = 0.95,
        temperature: float = 1.0,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Locally Typical Sampling

    [Intuition]:
    Improved version of typical sampling. Considers local properties of
    information content. Selects tokens whose information content is close
    to the expected value, avoiding extremes.

    [Formula]:
    .. math::
        |\\log P(x_t|x_{<t}) + H(P)| < \\text{threshold}

    where H(P) is the conditional entropy.

    Args:
        tau: Information tolerance (typically 0.9-0.99)
        temperature: Sampling temperature
    """
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    probs = torch.exp(log_probs)

    p_log_p = torch.nan_to_num(probs * log_probs, nan=0.0)
    entropy = -torch.sum(p_log_p, dim=-1, keepdim=True)

    shifted_entropy = torch.abs(-log_probs - entropy)

    sorted_entropy, sorted_indices = torch.sort(shifted_entropy, dim=-1)
    sorted_probs = torch.gather(probs, dim=-1, index=sorted_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > tau
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove
    )

    logits_masked = logits.masked_fill(indices_to_remove, float('-inf'))
    return multinomial_sampling(logits_masked, temperature=temperature)


def mirostat_sampling(
        logits: torch.Tensor,
        mu: torch.Tensor,
        tau: float = 5.0,
        eta: float = 0.1,
        temperature: float = 1.0,
        **kwargs
) -> tuple:
    """
    [Strategy]: Mirostat v2

    [Intuition]:
    Control text "surprisal" (perplexity) via a feedback loop. Dynamically
    adjusts how many tokens to consider (K) to maintain target surprisal tau.

    Think of it like cruise control for text randomness:
    - If text is too surprising: reduce K (be more conservative)
    - If text is too boring: increase K (be more creative)

    [Formula]:
    Surprisal of token x:

    .. math::
        s(x) = -\\log_2 P(x)

    Dynamic threshold:

    .. math::
        k = |\\{x : s(x) < \\mu + \\tau\\}|

    State update:

    .. math::
        \\mu_{t+1} = \\mu_t - \\eta (\\tau - s(x_t))

    [Implementation]:
    1. Compute current K based on surprisal threshold
    2. Sample from top K tokens
    3. Update mu (running estimate of surprisal) based on error

    Args:
        mu: Current average surprisal estimate (state variable)
        tau: Target surprisal level
        eta: Learning rate for mu updates
        temperature: Sampling temperature

    Returns:
        (next_token, updated_mu): Tuple of sampled token and new state
    """
    batch_size = logits.size(0)

    if temperature < 1e-5:
        temperature = 1.0

    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    sorted_surprise = -torch.log2(sorted_probs + 1e-8)

    next_tokens_list = []
    new_mus = mu.clone()

    for i in range(batch_size):
        # Dynamically determine K
        k_tensor = (sorted_surprise[i] < mu[i] + tau).long().sum()
        k = max(1, k_tensor.item())

        cand_probs = sorted_probs[i, :k]
        cand_probs_sum = cand_probs.sum()
        if cand_probs_sum > 0:
            cand_probs /= cand_probs_sum
        else:
            cand_probs = torch.tensor([1.0], device=logits.device)

        idx_in_cand = torch.multinomial(cand_probs, 1)
        token_idx = sorted_indices[i, idx_in_cand]
        next_tokens_list.append(token_idx)

        # Update mu using feedback
        observed_surprise = -torch.log2(probs[i, token_idx] + 1e-8)
        error = tau - observed_surprise
        new_mus[i] = mu[i] - eta * error

    next_tokens = torch.cat(next_tokens_list).view(batch_size, 1)
    return next_tokens, new_mus


# Combined annd Adaptive Strategies

def combined_top_k_top_p(
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Combined Top-K and Top-P

    [Intuition]:
    Get the best of both worlds:
    - Top-K provides a hard upper limit (efficiency, safety)
    - Top-P provides dynamic adaptation (quality)

    [Formula]:
    First apply Top-K:

    .. math::
        V^{(k)} = \\{x : P(x) \\ge P_{(k)}\\}

    Then apply Top-P on result:

    .. math::
        V^{(p)} = \\min\\{V' \\subseteq V^{(k)} : \\sum_{x \\in V'} P(x) \\ge p\\}

    Final set: :math:`V_{\\text{final}} = V^{(p)}`

    [Implementation]:
    Apply Top-K first, then Top-P on the result.

    Args:
        top_k: Maximum tokens to consider
        top_p: Cumulative probability threshold
        temperature: Sampling temperature
    """
    vocab_size = logits.size(-1)
    top_k = min(top_k, vocab_size)

    # Step 1: Top-K filtering
    top_k_values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold_k = top_k_values[:, -1].unsqueeze(-1)
    logits_k_masked = torch.where(
        logits < threshold_k,
        torch.tensor(float('-inf'), device=logits.device),
        logits
    )

    # Step 2: Top-P filtering on Top-K results
    sorted_logits, sorted_indices = torch.sort(logits_k_masked, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits / temperature, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove
    )

    logits_final = logits_k_masked.clone()
    logits_final[indices_to_remove] = float('-inf')

    return multinomial_sampling(logits_final, temperature=temperature)


def dynamic_temperature(
        logits: torch.Tensor,
        base_temperature: float = 1.0,
        min_temp: float = 0.5,
        max_temp: float = 1.3,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Dynamic Temperature

    [Intuition]:
    Adjust temperature based on the model's current confidence (entropy):
    - High entropy (uncertain) → Lower temperature (be conservative)
    - Low entropy (confident) → Can use higher temperature (explore more)

    [Formula]:
    .. math::
        T_{\\text{dynamic}} = T_{\\text{base}} \\times (2.0 - \\text{normalized_entropy})

    Args:
        base_temperature: Starting temperature
        min_temp: Lower bound
        max_temp: Upper bound
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    p_log_p = torch.nan_to_num(probs * log_probs, nan=0.0)
    entropy = -torch.sum(p_log_p, dim=-1, keepdim=True)

    max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float, device=logits.device))
    normalized_entropy = torch.clamp(entropy / (max_entropy + 1e-8), 0, 1)

    dynamic_temp = base_temperature * (2.0 - normalized_entropy)
    dynamic_temp = torch.clamp(dynamic_temp, min_temp, max_temp)

    return multinomial_sampling(logits, temperature=dynamic_temp.item())


def temperature_annealing(
        logits: torch.Tensor,
        step: int,
        max_steps: int,
        start_temp: float = 1.5,
        end_temp: float = 0.5,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Temperature Annealing

    [Intuition]:
    Like simulated annealing in optimization. Start with high temperature
    (explore widely), gradually reduce (converge to good solution).

    Useful for creative tasks where you want variety at first, then
    coherence at the end.

    [Formula]:
    .. math::
        T(t) = T_{\\text{start}} - (T_{\\text{start}} - T_{\\text{end}}) \\times \\frac{t}{T_{\\text{max}}}

    Args:
        step: Current generation step
        max_steps: Total generation steps
        start_temp: Initial temperature (high)
        end_temp: Final temperature (low)
    """
    step = min(step, max_steps)
    progress = step / max(max_steps, 1)
    current_temp = start_temp - (start_temp - end_temp) * progress
    current_temp = max(current_temp, 0.1)

    return multinomial_sampling(logits, temperature=current_temp)


def adaptive_sampling(
        logits: torch.Tensor,
        confidence_threshold: float = 0.8,
        high_conf_temp: float = 0.5,
        low_conf_temp: float = 1.2,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Adaptive Sampling

    [Intuition]:
    Choose sampling strategy based on model confidence:
    - High confidence (max_prob > threshold): Low temp (exploit)
    - Low confidence (max_prob < threshold): High temp (explore)

    [Formula]:
    .. math::
        T = \\begin{cases}
        T_{\\text{low}} & \\text{if } \\max_x P(x) > \\theta \\\\
        T_{\\text{high}} & \\text{otherwise}
        \\end{cases}

    where :math:`\\theta` is the confidence threshold.

    Args:
        confidence_threshold: Probability threshold for switching strategies
        high_conf_temp: Temperature when confident
        low_conf_temp: Temperature when uncertain
    """
    probs = F.softmax(logits, dim=-1)
    max_prob = torch.max(probs, dim=-1, keepdim=True)[0]

    # Decide temperature for each batch element
    batch_temps = []
    for i in range(max_prob.size(0)):
        if max_prob[i].item() > confidence_threshold:
            batch_temps.append(high_conf_temp)
        else:
            batch_temps.append(low_conf_temp)

    avg_temp = sum(batch_temps) / len(batch_temps)
    return multinomial_sampling(logits, temperature=avg_temp)


def contrastive_sampling(
        logits: torch.Tensor,
        sequences: torch.Tensor,
        penalty_alpha: float = 0.6,
        temperature: float = 1.0,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Contrastive Sampling (Simplified)

    [Intuition]:
    Balance between probability and degeneration (repetition).
    Penalize tokens that appear frequently in recent context.

    This is a simplified version. Full contrastive search needs hidden states.

    [Formula]:
    .. math::
        \\text{Score}(v) = (1-\\alpha) \\times \\log P(v) - \\alpha \\times \\text{RecentFreq}(v)

    Args:
        penalty_alpha: Contrastive penalty weight (0-1)
        temperature: Sampling temperature
    """
    penalty_alpha = max(0.0, min(1.0, penalty_alpha))
    batch_size = logits.size(0)
    logits_adjusted = logits.clone()

    window_size = 10

    for i in range(batch_size):
        seq = sequences[i]
        if seq.size(0) >= window_size:
            recent_tokens = seq[-window_size:]
            token_counts = torch.bincount(recent_tokens, minlength=logits.size(-1))
            token_counts = token_counts[:logits.size(-1)]

            frequency_penalty = token_counts.float() * penalty_alpha * 2.0
            logits_adjusted[i] -= frequency_penalty

    return multinomial_sampling(logits_adjusted, temperature=temperature)


def length_penalty_sampling(
        logits: torch.Tensor,
        current_length: int,
        target_length: int = 50,
        alpha: float = 1.0,
        temperature: float = 1.0,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Length Penalty Sampling

    [Intuition]:
    Encourage or discourage reaching a target length by adjusting the
    EOS token probability.

    - Near target length: Boost EOS probability
    - Far from target: Suppress EOS probability

    [Formula]:
    .. math::
        \\text{logit}_{\\text{EOS}} = \\text{logit}_{\\text{EOS}} + \\alpha \\times f(\\text{current/target})

    Args:
        current_length: Current sequence length
        target_length: Desired final length
        alpha: Penalty strength
        temperature: Sampling temperature
    """
    eos_id = 7  # Default from original code
    logits_adjusted = logits.clone()

    length_ratio = current_length / max(target_length, 1)

    if length_ratio >= 0.8:
        # Close to target: boost EOS
        boost = alpha * (length_ratio - 0.8) * 5.0
        logits_adjusted[:, eos_id] += boost
    elif length_ratio < 0.5:
        # Too short: suppress EOS
        penalty = alpha * (0.5 - length_ratio) * 3.0
        logits_adjusted[:, eos_id] -= penalty

    return multinomial_sampling(logits_adjusted, temperature=temperature)


# Penalty Mechanism

def apply_repetition_penalty(
        logits: torch.Tensor,
        sequences: torch.Tensor,
        penalty: float = 1.2,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Repetition Penalty

    [Intuition]:
    Penalize tokens that have already appeared in the generated sequence.
    Prevents models from getting stuck in loops like "the the the...".

    [Formula]:
    For all :math:`x \\in \\text{history}`:

    .. math::
        \\text{logit}(x)' = \\begin{cases}
        \\text{logit}(x) / \\text{penalty} & \\text{if } \\text{logit}(x) > 0 \\\\
        \\text{logit}(x) \\times \\text{penalty} & \\text{if } \\text{logit}(x) < 0
        \\end{cases}

    where penalty > 1.

    Args:
        logits: Current logits
        sequences: Previously generated tokens
        penalty: Penalty strength (typically 1.1-1.5)
    """
    batch_size = logits.shape[0]

    for i in range(batch_size):
        previous_tokens = sequences[i].unique()

        for token_id in previous_tokens:
            token_logit = logits[i, token_id]

            if token_logit < 0:
                logits[i, token_id] = token_logit * penalty
            else:
                logits[i, token_id] = token_logit / penalty

    return logits


def apply_frequency_presence_penalty(
        logits: torch.Tensor,
        sequences: torch.Tensor,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Frequency & Presence Penalty (OpenAI Style)

    [Intuition]:
    Additive penalties (not multiplicative like repetition_penalty):
    - Frequency: Penalize based on how often token appeared
    - Presence: Penalize based on whether token appeared at all

    [Formula]:
    .. math::
        \\text{logit}[x] = \\text{logit}[x] - (\\text{count}[x] \\times \\text{freq_p})
                           - (\\mathbb{1}_{\\text{exists}} \\times \\text{pres_p})

    Args:
        frequency_penalty: Per-occurrence penalty
        presence_penalty: Binary appearance penalty
    """
    batch_size = logits.shape[0]

    for i in range(batch_size):
        counts = torch.bincount(sequences[i], minlength=logits.shape[-1])
        counts = counts[:logits.shape[-1]]

        logits[i] -= counts * frequency_penalty
        logits[i] -= (counts > 0).float() * presence_penalty

    return logits


def apply_no_repeat_ngram(
        logits: torch.Tensor,
        sequences: torch.Tensor,
        ngram_size: int = 3,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: No-Repeat N-gram Blocking

    [Intuition]:
    Hard block on repeating n-grams. If history contains "the cat sat",
    and current context is "the cat", block "sat".

    Prevents exact phrase repetition.

    [Formula]:
    Let :math:`c = (x_{t-n+1}, ..., x_{t-1})` be the current context.

    Block token :math:`v` if:

    .. math::
        \\exists i < t : (x_i, x_{i+1}, ..., x_{i+n-2}) = c \\text{ and } x_{i+n-1} = v

    [Implementation]:
    1. Extract last (n-1) tokens as prefix
    2. Find all occurrences of this prefix in history
    3. Block the tokens that followed those occurrences

    Args:
        ngram_size: Size of n-gram to block (typically 2-4)
    """
    if ngram_size < 2:
        return logits

    batch_size = logits.size(0)
    logits_masked = logits.clone()

    for i in range(batch_size):
        seq = sequences[i]
        seq_len = seq.size(0)

        if seq_len < ngram_size - 1:
            continue

        current_prefix = seq[-(ngram_size - 1):].tolist()

        banned_tokens = set()
        for j in range(seq_len - ngram_size + 1):
            historical_prefix = seq[j:j + ngram_size - 1].tolist()

            if historical_prefix == current_prefix:
                next_token = seq[j + ngram_size - 1].item()
                banned_tokens.add(next_token)

        for token_id in banned_tokens:
            logits_masked[i, token_id] = float('-inf')

        # Safety: if all tokens banned, undo
        if torch.all(torch.isinf(logits_masked[i])):
            logits_masked[i] = logits[i]

    return logits_masked


def apply_encoder_repetition_penalty(
        logits: torch.Tensor,
        original_input: torch.Tensor,
        sequences: torch.Tensor,
        penalty: float = 1.5,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Encoder Repetition Penalty

    [Intuition]:
    Penalize not just generated tokens, but also tokens from the original prompt.
    Prevents model from "copying" the input.

    Useful for:
    - Summarization (don't just copy sentences)
    - Paraphrasing (must use different words)
    - Q&A (don't just repeat the question)

    [Formula]:
    Let :math:`S_{\\text{input}}` = original prompt tokens,
    :math:`S_{\\text{gen}}` = generated tokens.

    For all :math:`x \\in S_{\\text{input}} \\cup S_{\\text{gen}}`:

    .. math::
        \\text{logit}(x)' = \\begin{cases}
        \\text{logit}(x) / \\lambda & \\text{if } \\text{logit}(x) > 0 \\\\
        \\text{logit}(x) \\times \\lambda & \\text{if } \\text{logit}(x) < 0
        \\end{cases}

    where :math:`\\lambda > 1` is the penalty factor.

    Args:
        original_input: The initial prompt tokens
        sequences: Currently generated sequence
        penalty: Penalty strength
    """
    batch_size = logits.shape[0]
    logits_penalized = logits.clone()

    for i in range(batch_size):
        original_tokens = original_input[i].unique()
        generated_tokens = sequences[i].unique()
        all_tokens = torch.cat([original_tokens, generated_tokens]).unique()

        for token_id in all_tokens:
            token_logit = logits_penalized[i, token_id]

            if token_logit < 0:
                logits_penalized[i, token_id] = token_logit * penalty
            else:
                logits_penalized[i, token_id] = token_logit / penalty

    return logits_penalized



def apply_all_penalties(
        logits: torch.Tensor,
        sequences: torch.Tensor,
        original_input: torch.Tensor = None,
        **penalty_kwargs
) -> torch.Tensor:
    """
    Apply all configured penalties to logits in one place.
    This is used by beam search variants to ensure consistent penalty application.

    Args:
        logits: Current logits [batch_size, vocab_size]
        sequences: Generated sequence so far
        original_input: Original prompt (for encoder repetition penalty)
        **penalty_kwargs: All penalty parameters

    Returns:
        Penalized logits
    """
    # Apply frequency and presence penalties
    freq_p = penalty_kwargs.get('frequency_penalty', 0.0)
    pres_p = penalty_kwargs.get('presence_penalty', 0.0)
    if freq_p != 0.0 or pres_p != 0.0:
        logits = apply_frequency_presence_penalty(logits, sequences, freq_p, pres_p)

    # Apply repetition penalty
    rep_p = penalty_kwargs.get('repetition_penalty', 1.0)
    if rep_p != 1.0:
        logits = apply_repetition_penalty(logits, sequences, rep_p)

    # Apply encoder repetition penalty
    enc_rep_p = penalty_kwargs.get('encoder_repetition_penalty', 1.0)
    if enc_rep_p != 1.0 and original_input is not None:
        logits = apply_encoder_repetition_penalty(logits, original_input, sequences, enc_rep_p)

    # Apply no-repeat n-gram blocking
    ngram_size = penalty_kwargs.get('no_repeat_ngram_size', 0)
    if ngram_size > 0:
        logits = apply_no_repeat_ngram(logits, sequences, ngram_size)

    return logits


# Beam Search Variants

def beam_search_decode(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        max_new_tokens: int,
        beam_width: int = 3,
        eos_id: int = 7,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Beam Search

    [Intuition]:
    Instead of committing to one token at a time, maintain K candidate
    sequences and expand all of them. Like exploring multiple paths through
    a maze simultaneously.

    At each step:
    1. Expand each beam's next token (K * vocab_size options)
    2. Keep top K sequences by cumulative log probability

    More thorough than greedy, but K times slower.

    [Formula]:
    At each step t, maintain K beams :math:`B = \\{(s_1, y_1), ..., (s_K, y_K)\\}`
    where :math:`s_i` is the score and :math:`y_i` is the sequence.

    Score:

    .. math::
        s(y_{1:t}) = \\sum_{i=1}^{t} \\log P(y_i | y_{<i})

    Expansion:

    .. math::
        B_{t+1} = \\text{Top-K}_{(s,y) \\in B_t \\times V} \\left(s(y) + \\log P(v|y)\\right)

    Args:
        beam_width: Number of beams to maintain (typically 3-10)
    """
    print(f"--- Starting generation (strategy: beam_search, Width={beam_width}) ---")

    if inputs.shape[0] > 1:
        raise ValueError("Educational Beam Search implementation only supports batch_size=1")

    # Each beam is (score, sequence)
    beams = [(0.0, inputs)]

    for _ in range(max_new_tokens):
        candidates = []

        for score, seq in beams:
            if seq[0, -1].item() == eos_id:
                candidates.append((score, seq))
                continue

            idx_cond = (seq if seq.size(1) <= model.args.context_length
                        else seq[:, -model.args.context_length:])

            logits = model(idx_cond)
            next_token_logits = logits[:, -1, :]

            # Apply penalties
            next_token_logits = apply_all_penalties(
                next_token_logits,
                seq,
                original_input=inputs,
                **kwargs
            )

            log_probs = F.log_softmax(next_token_logits, dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_width, dim=-1)

            for k in range(beam_width):
                next_score = score + topk_log_probs[0, k].item()
                next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                next_seq = torch.cat((seq, next_token), dim=1)
                candidates.append((next_score, next_seq))

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        if all(seq[0, -1].item() == eos_id for _, seq in beams):
            break

    best_score, best_seq = beams[0]
    return best_seq


def diverse_beam_search(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        max_new_tokens: int,
        beam_width: int = 6,
        num_groups: int = 3,
        diversity_penalty: float = 0.5,
        eos_id: int = 7,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Diverse Beam Search

    [Intuition]:
    Regular beam search often produces very similar outputs.
    Divide beams into groups. Penalize groups for choosing tokens
    that previous groups chose. Forces diversity.

    [Formula]:
    .. math::
        \\text{Score}_{\\text{group}}(v) = \\log P(v) - \\lambda \\sum_{g'<g} \\mathbb{1}[v \\in \\text{tokens}_{g'}]

    Args:
        beam_width: Total beam width
        num_groups: Number of groups to divide beams into
        diversity_penalty: Penalty for choosing same tokens as other groups
    """
    print(f"--- Diverse Beam Search (Groups={num_groups}, Width={beam_width}) ---")

    if inputs.shape[0] > 1:
        raise ValueError("Diverse Beam Search currently supports batch_size=1")

    beams_per_group = max(1, beam_width // num_groups)
    all_groups = []

    for group_id in range(num_groups):
        group_beams = [(0.0, inputs)]

        for step in range(max_new_tokens):
            candidates = []

            for score, seq in group_beams:
                if seq[0, -1].item() == eos_id:
                    candidates.append((score, seq))
                    continue

                idx_cond = seq if seq.size(1) <= model.args.context_length else seq[:, -model.args.context_length:]
                logits = model(idx_cond)
                next_token_logits = logits[:, -1, :]

                # Apply penalties
                next_token_logits = apply_all_penalties(
                    next_token_logits,
                    seq,
                    original_input=inputs,
                    **kwargs
                )

                log_probs = F.log_softmax(next_token_logits, dim=-1)

                # Apply diversity penalty from previous groups
                if group_id > 0:
                    for prev_group in all_groups[:group_id]:
                        for _, prev_seq in prev_group:
                            if prev_seq.size(1) > inputs.size(1):
                                used_tokens = prev_seq[0, inputs.size(1):].unique()
                                for token in used_tokens:
                                    next_token_logits[0, token] -= diversity_penalty

                log_probs = F.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, k=min(beams_per_group, logits.size(-1)), dim=-1)

                for k in range(topk_log_probs.size(-1)):
                    next_score = score + topk_log_probs[0, k].item()
                    next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                    next_seq = torch.cat((seq, next_token), dim=1)
                    candidates.append((next_score, next_seq))

            group_beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beams_per_group]

            if all(seq[0, -1].item() == eos_id for _, seq in group_beams):
                break

        all_groups.append(group_beams)

    # Return best across all groups
    all_beams = [beam for group in all_groups for beam in group]
    best_score, best_seq = max(all_beams, key=lambda x: x[0])

    return best_seq


def constrained_beam_search(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        max_new_tokens: int,
        must_include: list = None,
        beam_width: int = 3,
        eos_id: int = 7,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Constrained Beam Search

    [Intuition]:
    Force generation to include certain tokens. Useful when you have
    requirements like "must mention 'climate change' in the summary".

    Track which constraints each beam has satisfied. Prioritize beams
    that satisfy more constraints.

    [Formula]:
    Let :math:`C = \\{c_1, c_2, ..., c_m\\}` be the set of required tokens.

    For each beam :math:`y`, track satisfied constraints:

    .. math::
        S(y) = \\{c \\in C : c \\in y\\}

    Beam ranking:

    .. math::
        \\text{rank}(y) = (|S(y)|, \\text{score}(y))

    Prioritize by number of satisfied constraints first, then by score.

    **Constraint Enforcement**:
    For unsatisfied constraints, boost their logits:

    .. math::
        \\text{logit}(c) = \\text{logit}(c) + \\beta \\quad \\text{if } c \\in C \\setminus S(y)

    where :math:`\\beta` is a large positive boost value.

    Args:
        must_include: List of token IDs that must appear
        beam_width: Beam width
    """
    print(f"--- Constrained Beam Search (Must include: {must_include}) ---")

    if inputs.shape[0] > 1:
        raise ValueError("Constrained Beam Search supports batch_size=1")

    if must_include is None:
        must_include = []

    # Convert to set for faster lookup
    constraint_set = set(must_include)

    # Boost value for unsatisfied constraints
    constraint_boost = 10.0

    # Each beam: (score, sequence, satisfied_constraints_set)
    beams = [(0.0, inputs, set())]

    for step in range(max_new_tokens):
        candidates = []

        for score, seq, satisfied in beams:
            if seq[0, -1].item() == eos_id:
                candidates.append((score, seq, satisfied))
                continue

            idx_cond = seq if seq.size(1) <= model.args.context_length else seq[:, -model.args.context_length:]
            logits = model(idx_cond)
            next_token_logits = logits[:, -1, :].clone()

            # Apply penalties (NEW!)
            next_token_logits = apply_all_penalties(
                next_token_logits,
                seq,
                original_input=inputs,
                **kwargs
            )

            # **KEY FIX**: Boost logits of unsatisfied constraints
            unsatisfied = constraint_set - satisfied
            if unsatisfied and len(satisfied) < len(must_include):
                for token_id in unsatisfied:
                    if token_id < next_token_logits.size(-1):  # Safety check
                        next_token_logits[0, token_id] += constraint_boost

            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Expand beam
            # Use larger K to ensure we don't miss boosted constraint tokens
            expand_k = max(beam_width, len(unsatisfied) + beam_width)
            topk_log_probs, topk_indices = torch.topk(log_probs, k=min(expand_k, log_probs.size(-1)), dim=-1)

            for k in range(topk_log_probs.size(-1)):
                next_token_id = topk_indices[0, k].item()
                next_score = score + topk_log_probs[0, k].item()
                next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                next_seq = torch.cat((seq, next_token), dim=1)

                # Update constraint satisfaction
                new_satisfied = satisfied.copy()
                if next_token_id in constraint_set:
                    new_satisfied.add(next_token_id)

                candidates.append((next_score, next_seq, new_satisfied))

        # Sort by (num_constraints_satisfied DESC, score DESC)
        candidates.sort(key=lambda x: (len(x[2]), x[0]), reverse=True)
        beams = candidates[:beam_width]

        if all(seq[0, -1].item() == eos_id for _, seq, _ in beams):
            break

    best_score, best_seq, best_satisfied = beams[0]

    if must_include:
        satisfied_count = len(best_satisfied)
        total_count = len(must_include)
        if satisfied_count < total_count:
            print(f"Warning: Only satisfied {satisfied_count}/{total_count} constraints")
            print(f"Satisfied: {best_satisfied}")
            print(f"Missing: {constraint_set - best_satisfied}")
        else:
            print(f"✓ All {total_count} constraints satisfied!")

    return best_seq


def stochastic_beam_search(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        max_new_tokens: int,
        beam_width: int = 3,
        temperature: float = 1.0,
        eos_id: int = 7,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Stochastic Beam Search

    [Intuition]:
    Mix beam search with randomness. Instead of always taking top-K,
    sample K tokens from a larger pool (e.g., top-2K).

    More diverse than deterministic beam search, but more coherent
    than pure sampling.

    [Formula]:
    Instead of deterministic selection:

    .. math::
        B_{t+1} = \\text{Top-K}(\\text{candidates})

    Use stochastic selection:

    .. math::
        B_{t+1} = \\text{Sample-K}(\\text{Top-2K}(\\text{candidates}), T)

    where sampling is proportional to :math:`\\exp(\\text{score}/T)`.

    Args:
        beam_width: Number of beams
        temperature: Sampling temperature for choosing among candidates
    """
    print(f"--- Stochastic Beam Search (Width={beam_width}, Temp={temperature}) ---")

    if inputs.shape[0] > 1:
        raise ValueError("Stochastic Beam Search supports batch_size=1")

    beams = [(0.0, inputs)]

    for _ in range(max_new_tokens):
        candidates = []

        for score, seq in beams:
            if seq[0, -1].item() == eos_id:
                candidates.append((score, seq))
                continue

            idx_cond = seq if seq.size(1) <= model.args.context_length else seq[:, -model.args.context_length:]
            logits = model(idx_cond)
            next_token_logits = logits[:, -1, :]

            # Apply penalties
            next_token_logits = apply_all_penalties(
                next_token_logits,
                seq,
                original_input=inputs,
                **kwargs
            )

            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Sample from top-2K instead of taking top-K
            candidate_k = min(beam_width * 2, logits.size(-1))
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, k=candidate_k, dim=-1)

            topk_probs = F.softmax(topk_log_probs, dim=-1)
            num_samples = min(beam_width, candidate_k)
            sampled_indices = torch.multinomial(topk_probs, num_samples=num_samples, replacement=False)

            for idx in sampled_indices[0]:
                next_score = score + topk_log_probs[0, idx].item()
                next_token = topk_indices[0, idx].unsqueeze(0).unsqueeze(0)
                next_seq = torch.cat((seq, next_token), dim=1)
                candidates.append((next_score, next_seq))

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        if all(seq[0, -1].item() == eos_id for _, seq in beams):
            break

    best_score, best_seq = beams[0]
    return best_seq


def best_of_n_sampling(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        max_new_tokens: int,
        n: int = 5,
        temperature: float = 0.8,
        eos_id: int = 7,
        **kwargs
) -> torch.Tensor:
    """
    [Strategy]: Best-of-N Sampling

    [Intuition]:
    Generate N different completions, then pick the best one.
    "Best" can mean:
    - Highest average log probability
    - Lowest perplexity
    - Highest score from a reward model (as in RLHF)

    We use average log probability here.

    This is computationally expensive (N times the work) but can
    significantly improve quality.

    [Formula]:
    Generate N sequences :math:`\\{y^{(1)}, y^{(2)}, ..., y^{(N)}\\}`.

    Score each by average log probability:

    .. math::
        \\text{score}(y^{(i)}) = \\frac{1}{|y^{(i)}|} \\sum_{t=1}^{|y^{(i)}|} \\log P(y^{(i)}_t | y^{(i)}_{<t})

    Select:

    .. math::
        y^* = \\arg\\max_{i \\in \\{1,...,N\\}} \\text{score}(y^{(i)})

    Args:
        n: Number of candidates to generate
        temperature: Sampling temperature for diversity
    """
    print(f"--- Best-of-{n} Sampling ---")

    n = max(1, n)
    candidates = []

    for candidate_id in range(n):
        sequence = inputs.clone()
        total_log_prob = 0.0
        token_count = 0

        for _ in range(max_new_tokens):
            idx_cond = sequence if sequence.size(1) <= model.args.context_length else sequence[
                :, -model.args.context_length:]
            logits = model(idx_cond)
            next_token_logits = logits[:, -1, :]

            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_prob = probs[0, next_token].item()
            token_log_prob = torch.log(torch.tensor(token_prob + 1e-10)).item()
            total_log_prob += token_log_prob
            token_count += 1

            sequence = torch.cat((sequence, next_token), dim=1)

            if next_token.item() == eos_id:
                break

        avg_log_prob = total_log_prob / max(token_count, 1)
        candidates.append((avg_log_prob, sequence, candidate_id))

    best_log_prob, best_seq, best_id = max(candidates, key=lambda x: x[0])
    print(f"Best candidate: #{best_id}, Avg log prob: {best_log_prob:.4f}")

    return best_seq

# Testing Part

if __name__ == "__main__":
    # Setup
    torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Wrapper('gpt2')
    eos_id = tokenizer.eos_token_id

    prompt_text = "Do not go gentle into that good night"
    inputs = tokenizer.encode(prompt_text, return_tensors='pt')

    print("=" * 60)
    print(f"Test with GPT-2")
    print(f"Prompt: '{prompt_text}'")
    print("=" * 60)

    def run_and_print(name, **kwargs):
        print(f"{name}")
        try:
            start_inputs = inputs.clone()
            output = generate(model, start_inputs, max_new_tokens=50, eos_id=eos_id, **kwargs)

            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            new_text = full_text[len(prompt_text):].strip()
            new_text = new_text.replace('\n', ' ')
            print(f"{new_text}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 40)

    # Group A: Basic Strategies
    print("\n--- Group A: Basic Strategies ---")
    run_and_print("1. Greedy", strategy='greedy')
    run_and_print("2. Temperature (T=0.7)", strategy='temperature', temperature=0.7)
    run_and_print("3. Top-K (K=40)", strategy='top_k', top_k=40)
    run_and_print("4. Top-P (P=0.9)", strategy='top_p', top_p=0.9)
    run_and_print("5. Min-P (P=0.05)", strategy='min_p', min_p=0.05)
    run_and_print("6. Epsilon (E=0.0002)", strategy='epsilon', epsilon=0.0002)

    # Group B: Information Theoretic
    print("\n--- Group B: Information Theoretic ---")
    run_and_print("7. Typical (Mass=0.2)", strategy='typical', mass=0.2)
    run_and_print("8. Locally Typical (Tau=0.5)", strategy='locally_typical', tau=0.5)
    run_and_print("9. Eta (Eta=0.002)", strategy='eta', eta=0.002)
    run_and_print("10. TFS (Z=0.95)", strategy='tfs', z=0.95)
    run_and_print("11. Top-A (A=0.5)", strategy='top_a', top_a=0.5)

    # Group C: Combined & Dynamic
    print("\n--- Group C: Combined & Dynamic ---")
    run_and_print("12. Top-K + Top-P", strategy='top_k_top_p', top_k=40, top_p=0.9)
    run_and_print("13. Dynamic Temp", strategy='dynamic_temperature')
    run_and_print("14. Temp Annealing", strategy='temperature_annealing', start_temp=1.5, end_temp=0.5)
    run_and_print("15. Adaptive", strategy='adaptive', confidence_threshold=0.8)
    run_and_print("16. Contrastive (Alpha=0.6)", strategy='contrastive', penalty_alpha=0.6)

    # Group D: Feedback Control
    print("\n--- Group D: Feedback Control ---")
    run_and_print("17. Mirostat", strategy='mirostat', tau=3.0, eta=0.1)
    run_and_print("18. Length Penalty (Alpha=1.0)", strategy='length_penalty', length_alpha=1.0, target_length=20)

    # Group E: Penalties
    print("\n--- Group E: Penalties ---")
    run_and_print("19. Repetition Penalty", strategy='greedy', repetition_penalty=1.2)
    run_and_print("20. Freq/Pres Penalty", strategy='greedy', frequency_penalty=0.5, presence_penalty=0.3)
    run_and_print("21. No-Repeat N-Gram", strategy='greedy', no_repeat_ngram_size=2)
    run_and_print("22. Encoder Rep Penalty", strategy='greedy', encoder_repetition_penalty=1.2)

    # Group F: Beam Search Variants
    # Beam Search must be used in conjunction with N-Gram penalties or repeated penalties
    # otherwise, GPT-2 will inevitably result in an infinite loop.
    print("\n--- Group F: Beam Search Variants ---")
    run_and_print("23. Beam Search (Width=3, Ngram=2)",
                  strategy='beam_search',
                  beam_width=3,
                  no_repeat_ngram_size=2,
                  repetition_penalty=1.2)

    run_and_print("24. Diverse Beam (Groups=2)",
                  strategy='diverse_beam_search',
                  beam_width=4,
                  num_groups=2,
                  diversity_penalty=0.5,
                  no_repeat_ngram_size=2)

    force_word = " learning"
    force_ids = tokenizer.encode(force_word, add_special_tokens=False)
    print(f"   (Constraint: Must include '{force_word}' -> IDs {force_ids})")
    run_and_print("25. Constrained Beam",
                  strategy='constrained_beam_search',
                  beam_width=5,
                  must_include=force_ids,
                  no_repeat_ngram_size=2,
                  repetition_penalty=1.2)

    run_and_print("26. Stochastic Beam",
                  strategy='stochastic_beam_search',
                  beam_width=3,
                  temperature=0.7,
                  no_repeat_ngram_size=2)

    run_and_print("27. Best-of-N (N=3)", strategy='best_of_n', n=3, temperature=0.8)

    # Group G: Batch Test
    print("\n--- Group G: Batch Processing ---")
    batch_prompts = ["The cat sat on", "The dog jumped over"]
    batch_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).input_ids

    print(f"28. Batch Inputs: {batch_prompts}")
    try:
        batch_out = generate(model, batch_inputs, max_new_tokens=10, strategy='top_p', top_p=0.9, eos_id=eos_id)
        print("Output 1:", tokenizer.decode(batch_out[0], skip_special_tokens=True))
        print("Output 2:", tokenizer.decode(batch_out[1], skip_special_tokens=True))
    except Exception as e:
        print(f"Batch Error: {e}")

    print("Finished! ")

