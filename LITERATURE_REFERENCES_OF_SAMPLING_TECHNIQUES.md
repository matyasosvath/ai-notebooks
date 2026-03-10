# Decoding Strategies for Neural Language Models

**A Technical Reference with Formulations and Citations**  
*February 2026*

This document provides a technical overview of the decoding strategies implemented in this project. For each method, we describe the underlying intuition, the mathematical formulation as originally presented in the source literature, and the corresponding citation. Strategies are grouped loosely by family. Several techniques — particularly Min-P, Tail Free Sampling, and Top-A — originated in open-source communities rather than academic publications; these are noted where relevant.

---

## Basic Sampling Strategies

### Greedy Decoding

At each generation step, greedy decoding selects the single token that maximises the conditional probability given the preceding context. There is no randomness whatsoever: the same input will always produce the same output. While this can seem attractive for its determinism and computational simplicity, the approach is well known to fall into degenerate repetition loops, producing output that is grammatically coherent but semantically stale. The problem is particularly acute in open-ended generation: once the model commits to a high-probability phrase, the conditional distribution at subsequent steps tends to reinforce the same choices, creating a feedback loop. Greedy decoding is generally considered unsuitable for creative or conversational generation, though it remains a useful baseline and performs reasonably well for shorter structured outputs such as translation.

**Formula:**
```
x_t = arg max_{v ∈ V} P(v | x_{<t})
```

**Reference:**  
Greedy decoding is a classical algorithm without a single canonical origin. Its application to neural sequence models is described, among many other places, in Graves (2012), *Sequence Transduction with Recurrent Neural Networks*, arXiv:1211.3711.  
https://arxiv.org/abs/1211.3711

---

### Temperature Sampling

Temperature sampling is perhaps the most conceptually fundamental technique in the family of stochastic decoders. The idea is borrowed directly from statistical physics: the "temperature" of a system controls how spread out its energy distribution is. Applied to a language model's output logits, dividing by a temperature T before the softmax sharpens or flattens the resulting probability distribution. When T < 1, the distribution becomes more peaked around the highest-probability tokens, making the model more conservative and predictable. When T > 1, the distribution flattens, giving lower-probability tokens a greater share of the sampling mass and producing more varied — if sometimes less coherent — output. At T = 1, the distribution is unchanged. Temperature is almost always used in conjunction with other truncation methods rather than on its own; applying temperature alone without truncation still leaves the full vocabulary available, meaning pathologically unlikely tokens can still be sampled with non-negligible frequency.

**Formula:**
```
P_i^temp = exp(ℓ_i/T) / Σ_j exp(ℓ_j/T)
```
where ℓ_i denotes the raw logit for token i and T > 0 is the temperature parameter.

**Reference:**  
The temperature concept in probabilistic modelling originates with Ackley, Hinton, and Sejnowski (1985), *A Learning Algorithm for Boltzmann Machines*, Cognitive Science 9(1), 147–169.  
DOI: [10.1207/s15516709cog0901_7](https://doi.org/10.1207/s15516709cog0901_7)

---

### Top-K Sampling

Top-K sampling was introduced as part of the story generation work of Fan et al. (2018) and quickly became a standard component in language model decoding pipelines. The method is simple: at each step, only the K most probable tokens are retained; all others have their logits set to −∞ before the softmax. The remaining K tokens are then renormalised and sampled from. The motivation is to avoid the incoherent output that results from sampling over the full vocabulary, where the long tail of low-probability tokens contributes disproportionate noise.

The weakness of Top-K is that the appropriate value of K is not stable across contexts. When the model is highly confident — say, predicting the continuation of an idiomatic phrase — the probability mass is concentrated in just a few tokens, and a large K admits many implausible candidates. Conversely, in highly ambiguous contexts, a small K artificially restricts diversity. This instability motivated the development of Top-P sampling.

**Formula:**  
Let V^(K) ⊂ V denote the K tokens with the highest probability. The adjusted distribution is:
```
P'(x) = P(x) / Σ_{y ∈ V^(K)} P(y)   if x ∈ V^(K)
P'(x) = 0                            otherwise
```

**Reference:**  
Fan, A., Lewis, M., and Dauphin, Y. (2018), *Hierarchical Neural Story Generation*, Proceedings of ACL 2018, pp. 889–898.  
DOI: [10.18653/v1/P18-1082](https://doi.org/10.18653/v1/P18-1082)  
arXiv: [1805.04833](https://arxiv.org/abs/1805.04833)  
Code: https://github.com/pytorch/fairseq

---

### Top-P Sampling (Nucleus Sampling)

Top-P sampling, introduced by Holtzman et al. (2020) under the name nucleus sampling, addresses the fixed-K limitation directly. Rather than choosing a fixed number of tokens, the method dynamically determines the candidate set at each step by accumulating tokens in descending probability order until their cumulative probability reaches a threshold p. The resulting set — the "nucleus" — contains exactly those tokens whose joint probability mass accounts for fraction p of the distribution. Everything outside the nucleus is masked.

The key insight is that this threshold adapts to the shape of the distribution. When the model is confident, the nucleus is small: a handful of tokens account for 90% of the mass. When the model is uncertain, the nucleus expands to include more options. This dynamic behaviour is what makes Top-P so effective in practice: it tends to sample high-quality tokens without artificially inflating or deflating diversity. Holtzman et al. provide extensive empirical evidence that sampling from the nucleus avoids both the repetitive degeneration of greedy and beam search methods and the incoherence of unrestricted sampling. Nucleus sampling has since become the de facto standard in modern large language model deployments.

**Formula:**  
Let V^(p) be the smallest set satisfying:
```
Σ_{x ∈ V^(p)} P(x) ≥ p
```
Sampling then proceeds from the renormalised distribution over V^(p).

**Reference:**  
Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2020), *The Curious Case of Neural Text Degeneration*, ICLR 2020.  
arXiv: [1904.09751](https://arxiv.org/abs/1904.09751)  
OpenReview: https://openreview.net/forum?id=rygGQyrFvH  
Official code: https://github.com/ari-holtzman/degen

---

### Min-P Sampling

Min-P sampling is a relatively recent community-developed alternative to Top-P that takes a different approach to adaptive truncation. Instead of setting a fixed cumulative mass threshold, the method defines a threshold relative to the most probable token in the current distribution. Specifically, any token whose probability falls below p_min × P_max (where P_max is the probability of the top token) is excluded from sampling.

The rationale is that when the model is very confident — i.e., when P_max is high — the threshold rises proportionally, cutting off a wider range of less-probable tokens. When the model is uncertain and P_max is low, the threshold lowers accordingly, admitting more diversity. Proponents argue this behaviour is more natural than Top-P in high-confidence situations, where nucleus sampling can sometimes include tokens that are several orders of magnitude less likely than the top token. Min-P does not have a formal academic paper; it emerged from discussions in the llama.cpp community around 2023 and was subsequently implemented in several open-source inference frameworks.

**Formula:**  
Let P_max = max_{x∈V} P(x). The retained vocabulary is:
```
V_kept = {x ∈ V | P(x) ≥ p_min · P_max}
```

**Reference:**  
Community-developed technique originating in the llama.cpp project.  
Primary discussion: https://github.com/ggerganov/llama.cpp/pull/3841 (2023).  
No formal academic publication exists.

---

## Information-Theoretic Sampling Methods

### Typical Sampling

Typical sampling, proposed by Meister et al. (2022), approaches the truncation problem from an information-theoretic standpoint. The key observation is that in natural human text, the per-token surprisal — the negative log probability − log P(x_t| x_{<t}) — tends to remain close to the conditional entropy of the distribution. Tokens that are either far more probable or far less probable than the "expected" surprisal are, in a sense, atypical of how the distribution actually functions in practice.

The method therefore selects the set of tokens whose surprisal is closest to the conditional entropy H(P), accumulating them until the mass of the selected set reaches a threshold parameter. Intuitively, this biases the decoder away from both highly predictable (boring) tokens and highly surprising (incoherent) ones, targeting the region of the distribution most consistent with natural language statistics. The approach is grounded in the information-theoretic notion of "typical sets" from coding theory, which describes the set of sequences that a source is likely to actually produce — as opposed to the sequences that individually maximise probability.

**Formula:**  
The conditional entropy is:
```
H(P) = −Σ_{x∈V} P(x) log P(x)
```
Tokens are ranked by their deviation from H(P):
```
δ(x) = |−log P(x) − H(P)|
```
and accumulated in order of increasing δ(x) until cumulative probability reaches mass τ.

**Reference:**  
Meister, C., Pimentel, T., Wiher, G., and Cotterell, R. (2022), *Typical Decoding for Natural Language Generation*, arXiv:2202.00666.  
Code: https://github.com/cimeister/typical-sampling

---

### Locally Typical Sampling

Locally typical sampling, published by Meister et al. (2023) in TACL, refines the typical sampling framework by making the information-theoretic criterion more precise. The earlier version of typical sampling treats the entropy of the full distribution as the reference point, but this can be a poor estimate of the "local" information content at a given position in the sequence. The locally typical variant addresses this by conditioning the entropy computation on the current context more carefully and framing the problem in terms of conditional information content rather than marginal entropy.

The practical difference is subtle but meaningful: locally typical sampling tends to be better calibrated in situations where the model's distribution is highly context-dependent, which is of course virtually all situations in open-ended text generation. The formal property being enforced is that the decoder should prefer tokens whose self-information is close to the conditional entropy, consistent with the definition of a typical sequence from information theory.

**Formula:**
```
|log P(x_t| x_{<t}) + H(P(· | x_{<t}))| < threshold
```
Tokens satisfying this condition are collected until the cumulative probability reaches a preset mass τ.

**Reference:**  
Meister, C., Wiher, G., Pimentel, T., and Cotterell, R. (2023), *Locally Typical Sampling*, Transactions of the Association for Computational Linguistics (TACL), Vol. 11, pp. 102–121.  
DOI: [10.1162/tacl_a_00536](https://doi.org/10.1162/tacl_a_00536)  
ACL Anthology: https://aclanthology.org/2023.tacl-1.6/

---

### Eta Sampling (η-sampling)

Eta sampling was introduced by Hewitt, Manning, and Liang (2022) as part of a broader study on truncation methods framed through the lens of language model "desmoothing." The paper argues that language models, during training, are forced to assign some probability to every token in the vocabulary due to the softmax, even tokens that are clearly inappropriate in context. This artificial smoothing creates an unreliable tail in the distribution that truncation methods should systematically remove.

The η-sampling method computes a dynamic cutoff based on the entropy of the distribution. When the model is highly uncertain (high entropy), a looser cutoff is applied; when the model is confident (low entropy), the cutoff tightens. The specific formula takes the minimum of a fixed baseline η and a term that grows with entropy, ensuring the cutoff never drops below a meaningful floor while still adapting to distribution shape. The paper shows this adaptive behaviour is theoretically principled and empirically outperforms fixed-threshold methods on several generation benchmarks.

**Formula:**
```
ϵ = min(η, √(η · H(p)))
```
Tokens with P(x) < ϵ are removed from the candidate set.

**Reference:**  
Hewitt, J., Manning, C.D., and Liang, P. (2022), *Truncation Sampling as Language Model Desmoothing*, Findings of EMNLP 2022.  
arXiv: [2210.15191](https://arxiv.org/abs/2210.15191)  
Code: https://github.com/john-hewitt/truncation-sampling

---

### Tail Free Sampling (TFS)

Tail free sampling (TFS) takes a different geometric approach to the truncation problem. Rather than examining the probability values directly, it examines the rate of change of the probability curve. The intuition is that the probability distribution over the vocabulary typically has a prominent head (a few very likely tokens), a moderate body, and a long, flat tail. The "tail" starts at the inflection point where the curve transitions from meaningful probability differences to negligible ones.

TFS approximates this inflection point using a discrete second derivative of the sorted probability sequence. Where the second derivative is large, the curve is bending rapidly and we are still in the informative head region. Where it flattens out, we have entered the tail. The method accumulates tokens in probability-descending order, stopping when the normalised cumulative second derivative exceeds a threshold z. This is purely a practical heuristic developed by practitioners in the open-source text generation community; no formal mathematical justification has been published.

**Formula:**  
Let sorted probabilities be p_1 ≥ p_2 ≥ ... ≥ p_n. Define the first and second discrete derivatives:
```
d^(1)_i = p_i − p_{i+1}
d^(2)_i = d^(1)_i − d^(1)_{i+1}
```
Retain tokens 1, ..., k where k is the smallest index satisfying:
```
Σ_{i=1}^k |d^(2)_i| / Σ_j |d^(2)_j| > z
```

**Reference:**  
Community-developed heuristic, c. 2021. Attributed to discussions in the AI Dungeon and KoboldAI communities.  
Explanatory blog post: https://www.trentonbricken.com/Tail-Free-Sampling/  
No academic publication exists.

---

### Top-A Sampling

Top-A sampling is a community heuristic that defines a probability threshold relative to the square of the most probable token's probability. The mechanism is simpler than Min-P: the cutoff is set to a · P²_max, meaning tokens below this value are excluded. The quadratic dependence on P_max causes the filter to tighten aggressively when the top token is very confident and to relax quickly when the distribution is more diffuse. Like several other community samplers, Top-A lacks formal derivation or empirical justification in any peer-reviewed venue.

**Formula:**
```
V_kept = {x ∈ V | P(x) ≥ a · P²_max}
```

**Reference:**  
Community-developed technique, origin attributed to open-source text generation forums c. 2020–2021.  
No academic publication exists.

---

### Epsilon Sampling

Epsilon sampling is the simplest of the truncation methods: every token whose probability falls below an absolute threshold ε is excluded from sampling. Unlike Min-P (which scales the threshold to the top token's probability) or η-sampling (which adapts the threshold to entropy), epsilon sampling uses a fixed cutoff. This makes it easy to reason about but potentially brittle: the same ε may be too tight when the distribution is peaky and too loose when it is flat. It is primarily useful as a conservative safety net to remove clearly implausible tokens, and is most often employed in combination with other methods.

**Formula:**
```
V_kept = {x ∈ V | P(x) ≥ ε}
```

**Reference:**  
Epsilon sampling is discussed as a baseline truncation strategy in Hewitt et al. (2022); see the η-sampling reference above.

---

## Feedback-Controlled Decoding

### Mirostat v2

Mirostat, introduced by Basu et al. (2021), is perhaps the most mechanistically novel of all the methods described here. Rather than choosing a static parameter before generation begins, Mirostat maintains a running state µ and continuously updates it during generation to keep the observed surprisal of each selected token close to a user-specified target τ. This is analogous to a control loop: if the most recently sampled token was more surprising than intended, the system relaxes the threshold (allowing more candidates), and if the token was less surprising than intended, it tightens the threshold. The result is a decoding process that actively regulates the information content of the generated text, preventing the low-perplexity loops associated with greedy methods while also preventing the incoherence of high-temperature sampling.

The name Mirostat plays on the idea of a thermostat maintaining temperature: it maintains a target surprisal level throughout generation. This is particularly useful for long-form generation where one wants consistent fluency throughout rather than intermittent incoherent bursts. In contrast to all other methods in this document, Mirostat is stateful: the parameter µ evolves over the course of generation and must be carried forward from step to step.

**Formula:**  
The surprisal of a sampled token x is:
```
s(x) = −log₂ P(x)
```
At each step, the candidate set is defined as:
```
V^(mir) = {x ∈ V | s(x) < µ + τ}
```
After sampling token x_t, the state is updated via:
```
µ_{t+1} = µ_t − η(τ − s(x_t))
```
where η > 0 is the learning rate controlling how quickly the system responds to deviations from the target surprisal.

**Reference:**  
Basu, S., Ramachandran, G.S., Keskar, N.S., and Varshney, L.R. (2021), *Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity*, ICLR 2021.  
arXiv: [2007.14966](https://arxiv.org/abs/2007.14966)  
OpenReview: https://openreview.net/forum?id=W1G1JZEIy5_  
Official code: https://github.com/basusourya/mirostat

---

## Penalty and Constraint Mechanisms

### Repetition Penalty

Repetition penalty is a logit-level post-processing step applied before sampling, designed to suppress tokens that have already appeared in the generated sequence. The intuition is straightforward: if a token has been used recently, its logit should be deflated to make it less likely to be sampled again. The method was formalised in the context of the CTRL conditional language model, where controllable generation made repetition particularly problematic.

The adjustment is multiplicative rather than additive, applied asymmetrically depending on the sign of the logit. For tokens with positive logits (inherently more probable), the logit is divided by the penalty factor, thereby reducing the probability. For tokens with negative logits (inherently less probable), the logit is multiplied by the penalty, pushing it further negative. This asymmetric treatment ensures the penalty has a meaningful effect regardless of the logit's sign.

**Formula:**  
For each token x appearing in the generation history:
```
ℓ'(x) = ℓ(x) / θ   if ℓ(x) > 0
ℓ'(x) = ℓ(x) · θ   if ℓ(x) ≤ 0
```
where θ > 1 is the penalty strength and ℓ(x) is the raw logit.

**Reference:**  
Keskar, N.S., McCann, B., Varshney, L.R., Xiong, C., and Socher, R. (2019), *CTRL: A Conditional Transformer Language Model for Controllable Generation*, arXiv:1909.05858.  
Code: https://github.com/salesforce/ctrl

---

### Frequency and Presence Penalty

Frequency and presence penalties are additive penalties introduced by OpenAI in the context of the GPT-3 API. They differ from repetition penalty in two ways: they are subtractive (applied by subtraction from the logit rather than division) and they distinguish between two notions of repetition. The frequency penalty scales with the number of times a token has appeared, making each additional occurrence progressively less likely. The presence penalty is binary: it applies a fixed deduction to any token that has appeared at all, regardless of how many times.

In practice, frequency penalty tends to produce more varied vocabulary throughout a passage, while presence penalty introduces a one-time discouragement that prevents the model from immediately reusing recently introduced vocabulary. Both can be applied simultaneously.

**Formula:**
```
ℓ'(x) = ℓ(x) − (count(x) · α_freq) − (𝟙[x ∈ history] · α_pres)
```
where count(x) is the number of times token x has appeared in the generated sequence.

**Reference:**  
OpenAI API documentation, GPT-3 (2020).  
https://platform.openai.com/docs/api-reference/completions  
No formal academic paper; this is an industry technique.

---

### No-Repeat N-gram Blocking

While repetition penalty reduces the probability of previously used tokens in a soft manner, n-gram blocking enforces a hard constraint: if the last n − 1 tokens match the beginning of any n-gram that has already appeared in the sequence, the completing token is assigned a logit of −∞ and can never be sampled. This completely prevents exact phrase repetition, which is a common failure mode in abstractive summarisation where the model tends to copy sentences verbatim from the input.

The technique was introduced in the pointer-generator network work of See et al. (2017), originally in the form of a coverage mechanism for avoiding repetition in summarisation. The n-gram blocking variant became widely used in subsequent work and is now a standard option in most decoding libraries.

**Formula:**  
Let c = (x_{t−n+1}, ..., x_{t−1}) denote the current context window of length n − 1. Token v is blocked if:
```
∃ i < t : (x_i, x_{i+1}, ..., x_{i+n−2}) = c and x_{i+n−1} = v
```

**Reference:**  
See, A., Liu, P.J., and Manning, C.D. (2017), *Get To The Point: Summarization with Pointer-Generator Networks*, ACL 2017.  
arXiv: [1704.04368](https://arxiv.org/abs/1704.04368)  
DOI: [10.18653/v1/P17-1099](https://doi.org/10.18653/v1/P17-1099)  
Code: https://github.com/abisee/pointer-generator

---

### Encoder Repetition Penalty

Encoder repetition penalty is an extension of the standard repetition penalty that additionally penalises tokens appearing in the original input prompt, not just in the generated continuation. The motivation is to discourage the model from essentially paraphrasing the input back at the user — a common failure mode in summarisation, question answering, and paraphrase generation tasks. By treating the prompt tokens as already-used vocabulary, the decoder is pushed toward generating text that expresses new content rather than echoing the input. The penalty formula is identical to the standard repetition penalty, but the set of penalised tokens is expanded to include the union of the input tokens and the generated tokens.

---

### Length Penalty Sampling

Length penalty is a mechanism for biasing the decoder toward or away from producing sequences of a target length. It operates by boosting the logit of the EOS (end-of-sequence) token when the sequence is approaching a desired length and suppressing it when the sequence is still short. This avoids the common problem of autoregressive models terminating too early on simple prompts or generating indefinitely on complex ones.

Length penalty in sampling is distinct from the length normalisation used in beam search (see Section below), which normalises cumulative scores by sequence length during the search process. Here, the penalty acts directly on the probability of the EOS token at each sampling step.

**Reference:**  
Length normalisation for beam search is described in Wu et al. (2016), *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*, arXiv:1609.08144.

---

## Beam Search Variants

### Beam Search

Beam search is the standard workhorse of deterministic sequence generation. Rather than committing to a single token at each step (as in greedy decoding), beam search maintains a set of K candidate sequences (beams) simultaneously. At each step, every beam is expanded over the full vocabulary, generating K × |V| candidate extensions, and only the top-K extensions by cumulative log probability are retained. This allows the decoder to explore multiple hypotheses in parallel and to recover from locally suboptimal choices that greedy decoding would be trapped by.

Beam search is guaranteed to find a better sequence than greedy decoding given the same scoring function, and it is substantially more thorough. However, it is well documented that the sequences it returns often suffer from the same repetition and blandness as greedy output, only to a lesser degree — the highest-cumulative-log-probability sequence is often not the highest-quality one. This is sometimes described as "the beam search curse," and it motivates many of the more sophisticated variants described in the following subsections. Beam search is also K times more computationally expensive than greedy decoding.

**Formula:**  
Maintaining K beams B = {(s_i, y_i)}^K_{i=1}, the score of a partial sequence y_{1:t} is:
```
s(y_{1:t}) = Σ^t_{i=1} log P(y_i| y_{<i})
```
At each step:
```
B_{t+1} = Top-K_{(s,y) ∈ B_t×V} [s(y) + log P(v | y)]
```

**Reference:**  
Beam search originates in speech recognition (Lowerre, 1976). Its application to neural sequence-to-sequence models is described in Sutskever, I., Vinyals, O., and Le, Q.V. (2014), *Sequence to Sequence Learning with Neural Networks*, NIPS 2014.  
arXiv: [1409.3215](https://arxiv.org/abs/1409.3215)

---

### Diverse Beam Search

Standard beam search suffers from a well-known lack of output diversity: the K beams typically converge to very similar sequences, since they all maximise the same scoring function. Diverse beam search (DBS) addresses this by partitioning the beams into G groups and introducing a diversity penalty that discourages beams in later groups from repeating token choices made by earlier groups.

The groups are processed sequentially at each decoding step. When group g is being expanded, the log probabilities for all tokens already selected by groups 1, ..., g − 1 at this step are penalised by a term λ. This causes each group to effectively explore a different region of the output space, producing a diverse set of completions rather than a set of nearly-identical high-probability sequences. DBS is particularly valuable in applications such as image captioning or open-ended question answering, where multiple distinct, plausible outputs are desirable.

**Formula:**  
The score for group g selecting token v at step t is:
```
s̃_g(v) = log P(v) − λ Σ_{g'<g} 𝟙[v ∈ T_{g',t}]
```
where T_{g',t} is the set of tokens chosen by group g' at step t.

**Reference:**  
Vijayakumar, A.K., Cogswell, M., Selvaraju, R.R., Sun, Q., Lee, S., Crandall, D.J., and Batra, D. (2018), *Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models*, AAAI 2018.  
arXiv: [1610.02424](https://arxiv.org/abs/1610.02424)  
DOI: [10.1609/aaai.v32i1.12340](https://doi.org/10.1609/aaai.v32i1.12340)  
Code: https://github.com/ashwinkalyan/dbs

---

### Constrained Beam Search

Constrained beam search modifies the standard beam search objective to enforce that certain tokens must appear in the final output. This is useful in controlled generation tasks where the output is required to include specific keywords, named entities, or concepts. The approach used here maintains, for each beam, a record of which constraints have been satisfied so far, and boosts the logit of unsatisfied constraint tokens at each step to increase the probability that the beam will eventually satisfy them. Beams are ranked by the number of constraints satisfied before falling back to the score criterion, ensuring constraint satisfaction takes precedence over raw likelihood.

The approach is related to Grid Beam Search (Hokamp and Liu, 2017), which provides a more rigorous framework for lexically constrained decoding. An alternative formulation by Post and Vilar (2018) offers improved computational efficiency for practical deployment.

**Formula:**  
Let C = {c_1, ..., c_m} be the set of required tokens and S(y) = {c ∈ C : c ∈ y} the constraints satisfied by beam y. Beams are ranked lexicographically by:
```
rank(y) = (|S(y)|, s(y))
```

**References:**  
Hokamp, C. and Liu, Q. (2017), *Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search*, ACL 2017.  
arXiv: [1704.05906](https://arxiv.org/abs/1704.05906)  
DOI: [10.18653/v1/P17-1141](https://doi.org/10.18653/v1/P17-1141)

Post, M. and Vilar, D. (2018), *Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation*, NAACL 2018.  
arXiv: [1804.06189](https://arxiv.org/abs/1804.06189)

---

### Stochastic Beam Search

Stochastic beam search injects randomness into the beam expansion step in a principled way, drawing on the Gumbel-top-k trick from combinatorics. Rather than deterministically selecting the top-K tokens at each expansion, the method samples K tokens from a larger candidate pool without replacement, with probabilities proportional to their exponentiated scores. This introduces diversity into the beam set naturally, avoiding the convergence to near-identical sequences that afflicts standard beam search while retaining the structured exploration of multiple hypotheses. The theoretical justification, detailed in Kool et al. (2019), connects this sampling procedure to exact sampling from a distribution over sequences without replacement.

**Formula:**
```
B_{t+1} = Sample-K(Top-2K(candidates), T)
```
where sampling is proportional to exp(score/T).

**Reference:**  
Kool, W., van Hoof, H., and Welling, M. (2019), *Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement*, ICML 2019.  
arXiv: [1903.06059](https://arxiv.org/abs/1903.06059)

---

## Multi-Candidate and Adaptive Methods

### Best-of-N Sampling

Best-of-N sampling is a computationally expensive but conceptually simple method: generate N independent completions from the model, score each one, and return the best. In the simplest case, "best" is defined by the average log probability of the generated tokens — the sequence with the lowest per-token perplexity. In more sophisticated deployments, a separate reward model is used to score each completion, selecting the one that best satisfies some external criterion such as human preference or task-specific quality.

This method is most closely associated with the reinforcement learning from human feedback (RLHF) literature, where best-of-N sampling serves both as a practical inference-time improvement technique and as a baseline for evaluating reward model quality. Since a reward model should be able to identify the best completion from a set, its performance on best-of-N selection is a natural diagnostic. The method scales poorly with N (it requires N full forward passes), but for moderate N it can significantly improve output quality compared to single-pass sampling.

**Formula:**
```
y* = arg max_{i∈{1,...,N}} (1/|y^(i)|) Σ^{|y^(i)|}_{t=1} log P(y^(i)_t | y^(i)_{<t})
```

**Reference:**  
Ouyang, L. et al. (2022), *Training Language Models to Follow Instructions with Human Feedback*, NeurIPS 2022.  
arXiv: [2203.02155](https://arxiv.org/abs/2203.02155)  
Best-of-N is used as an evaluation baseline throughout this paper.

---

### Contrastive Search

Contrastive search, proposed by Su et al. (2022), takes a fundamentally different approach from all the truncation-based methods described above. The key insight is that repetition in language model outputs can be understood as a failure of the model's internal representations to remain diverse: when a model falls into a repetitive loop, the hidden states at successive positions become nearly identical. Contrastive search explicitly penalises this by scoring each candidate token not only by its probability but also by its cosine similarity to the hidden states of all previously generated tokens. Tokens that are highly similar to recent context representations are penalised, discouraging repetition at the level of latent semantics rather than surface vocabulary.

**Formula:**
```
s̃(v) = (1 − α) · log P(v | x_{<t}) − α · max_{1≤j<t} cos(h_v, h_{x_j})
```
where h_v is the hidden state produced when token v is appended, α ∈ [0, 1] balances the two objectives, and cos(·, ·) denotes cosine similarity. Our implementation uses a simplified approximation based on token-frequency penalties, since full contrastive search requires access to intermediate model representations not exposed by our wrapper.

**Reference:**  
Su, Y., Lan, T., Wang, Y., Yogatama, D., Kong, L., and Collier, N. (2022), *A Contrastive Framework for Neural Text Generation*, NeurIPS 2022.  
arXiv: [2202.06417](https://arxiv.org/abs/2202.06417)  
Code: https://github.com/yxuansu/SimCTG

---

### Dynamic Temperature

Dynamic temperature is a heuristic extension of temperature sampling in which the temperature is computed from the entropy of the current distribution at each step rather than fixed as a hyperparameter. The motivation is that a single temperature may be too high for confident steps (causing incoherent choices from a narrow distribution) and too low for uncertain steps (failing to exploit the breadth of valid options). By scaling the temperature inversely with the normalised entropy, this method applies conservative sampling when the model is confident and more exploratory sampling when it is not.

**Formula:**
```
T_dyn = T_base · (2.0 − Ĥ(P)), where Ĥ(P) = H(P) / log |V|
```
where Ĥ(P) ∈ [0, 1] is the normalised entropy, and the result is clamped to [T_min, T_max].

**Reference:**  
A practical heuristic without a canonical academic source.

---

### Temperature Annealing

Temperature annealing applies the concept of simulated annealing from combinatorial optimisation to the text decoding process. The temperature begins high at the start of generation, encouraging exploration of diverse output directions, and is linearly reduced toward a lower value as generation proceeds, consolidating toward a coherent, high-probability continuation. This can be useful for creative writing tasks where variety is desired early in a passage but consistency is important toward the end.

**Formula:**
```
T(t) = T_start − (T_start − T_end) · t/t_max
```
where t is the current generation step and t_max is the total number of steps.

**Reference:**  
Simulated annealing originates in Kirkpatrick, Gelatt, and Vecchi (1983), *Optimization by Simulated Annealing*, Science 220(4598), 671–680.  
DOI: [10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671)  
The application to decoding temperature is a practical adaptation.

---

### Adaptive Sampling

Adaptive sampling is a simple confidence-gated strategy: if the model's top token has probability exceeding a threshold θ, a low temperature is applied; otherwise, a high temperature is used. The reasoning is that high model confidence (a sharply peaked distribution) calls for exploitation of the most likely tokens, while uncertainty calls for broader exploration. This is a coarser version of the idea underlying dynamic temperature, applied as a binary switch rather than a continuous function.

**Formula:**
```
T = T_low    if P_max > θ
T = T_high   otherwise
```

**Reference:**  
A practical heuristic with no formal academic origin.

---

## Summary

This document covers 27 distinct decoding strategies for neural language models, ranging from fundamental approaches like greedy decoding and temperature sampling to sophisticated variants like Mirostat and contrastive search. Each method represents a different trade-off between quality, diversity, computational cost, and controllability.

The techniques can be broadly categorized as:
- **Basic sampling** (greedy, temperature, top-k, top-p, min-p)
- **Information-theoretic** (typical, locally typical, eta, TFS, top-a, epsilon)
- **Feedback-controlled** (Mirostat)
- **Penalty mechanisms** (repetition, frequency/presence, n-gram blocking)
- **Beam search variants** (standard, diverse, constrained, stochastic)
- **Multi-candidate methods** (best-of-N, contrastive search)
- **Adaptive approaches** (dynamic temperature, annealing, adaptive)

For implementation details and code examples, please refer to the project repository.
