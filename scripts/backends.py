#!/usr/bin/env python3
"""One backend per model: the loaded weights, its tokenizer, and where its blocks live.

WHY THIS FILE EXISTS. Two jobs reach into a model the same way. Extraction registers a
forward hook to CAPTURE a hidden state; steered generation registers one to MODIFY it.
Both need `blocks` -- `model.transformer.h` for CrystaLLM against `model.model.layers`
for LLaMA -- and that single line is the whole of the model-specific surface they share.
Written twice, the two copies drift. So the backends live here, beside utils.py /
manifold.py / predictors.py, and both scripts import them.

What is NOT here: what text a structure or a prompt becomes. That is a separate axis --
the same LLaMA weights can be fed a raw CIF or a crystal string, and it is the TEXT that
decides what a pooled vector or a generated sample means. Extraction keeps its text
builders; generation keeps its prompt sources.

    method                                      used by
    blocks, n_layer, n_embd, block_size, pad_id both
    encode_pair(prompt, answer), forward(ids)   extraction
    encode(text), decode(ids)                   both
    generate(prompt, ...)                       generation
    hidden_mean(text, layer)                    generation (pca_local)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MODELS


def load_model(model_dir: str, device: torch.device):
    """Load CrystaLLM from a nanoGPT-style ckpt.pt. Returns (model, config).

    Kept at this name and signature because steer_generate_cif.py, test_kv_cache.py,
    layer_causal_probe.py, layernorm_survival.py, manifold_distance.py,
    injection_magnitude.py and analyze_steering_norms.py all import it -- from
    extract_cif_embeddings.py, which re-exports it from here.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CrystaLLM"))
    from crystallm import GPTConfig, GPT

    ckpt = torch.load(os.path.join(model_dir, "ckpt.pt"), map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    # Disable dropout for inference. The checkpoint ships dropout=0.1, and the functional
    # SDPA dropout_p (_model.py) is NOT gated by model.eval(), so leaving it on drops ~10%
    # of attention weights on every forward -> nondeterministic generation. This loader is
    # inference-only, so force dropout to 0 here (covers SDPA + all nn.Dropout modules).
    config.dropout = 0.0
    model = GPT(config)
    state_dict = ckpt["model"]
    # strip compile prefix if present
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded model: {config.n_layer} layers, {config.n_embd} dim, block_size {config.block_size}")
    return model, config


class Backend:
    """Shared behaviour. Subclasses supply the model, tokenizer, blocks and sampling."""

    def hidden_mean(self, text: str, layer: int):
        """Mean-pooled layer-L hidden state of `text`, as a float32 (n_embd,) tensor.

        Used to place a prompt in the PCA subspace before generation starts. Note this
        pools a PROMPT while the corpus embeddings pooled a whole structure -- the only
        view available before any token has been generated.
        """
        captured = {}

        def grab(module, inp, out):
            captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        handle = self.blocks[layer].register_forward_hook(grab)
        x = torch.tensor(self.encode(text), dtype=torch.long, device=self.device).unsqueeze(0)
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            handle.remove()
        return captured["h"][0].float().mean(0)


class CrystaLLMBackend(Backend):
    """CrystaLLM v1: nanoGPT blocks at model.transformer.h, CIFTokenizer."""

    def __init__(self, ckpt_dir: str, device: torch.device, torch_dtype: str = None):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CrystaLLM"))
        from crystallm import CIFTokenizer

        self.model, config = load_model(ckpt_dir, device)
        self.tokenizer = CIFTokenizer()
        self.device = device
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.blocks = self.model.transformer.h
        self.pad_id = 0
        self.config = config

    def encode(self, text: str):
        return self.tokenizer.encode(self.tokenizer.tokenize_cif(text))

    def decode(self, ids):
        return self.tokenizer.decode(list(ids))

    def encode_pair(self, prompt: str, answer: str):
        """(ids, answer_start). CrystaLLM has no prompt form, so prompt must be empty."""
        assert prompt == "", "the CrystaLLM tokenizer has no prompt/answer split"
        return self.encode(answer)[:self.block_size], 0

    def forward(self, input_ids):
        self.model(input_ids)

    def generate_ids(self, prompt: str, max_new_tokens: int, temperature: float = 1.0,
                     top_k: int = None, top_p: float = None, use_cache: bool = True):
        """((1, T) token ids, prompt length). top_p is accepted and ignored -- nanoGPT
        samples with top_k only, and silently switching sampler would change every
        existing result."""
        ids = self.encode(prompt)
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            gen = self.model.generate_cached if use_cache else self.model.generate
            return gen(x, max_new_tokens, temperature=temperature, top_k=top_k), len(ids)

    def generate(self, prompt: str, max_new_tokens: int, **kw) -> str:
        """The whole sequence as text, PROMPT INCLUDED.

        CrystaLLM is prompted with the head of a CIF and continues it, so the prompt is
        part of the structure it produced; cutting it off would leave a CIF with no
        formula line.
        """
        y, _ = self.generate_ids(prompt, max_new_tokens, **kw)
        return self.decode(y[0].tolist())


class LlamatBackend(Backend):
    """LLaMat-2: HuggingFace LLaMA-2, blocks at model.model.layers, BPE tokenizer.

    Loaded in half precision by default -- a 7B model is ~27 GB in float32 and ~13 GB in
    float16, and only the latter leaves room for activations on a 32 GB card. bfloat16
    needs sm_80 (Ampere); the V100s here are sm_70, so float16 is the default.
    """

    def __init__(self, ckpt_dir: str, device: torch.device, torch_dtype: str = "float16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, torch_dtype=getattr(torch, torch_dtype), low_cpu_mem_usage=True)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        config = self.model.config
        self.config = config
        self.device = device
        self.n_layer = config.num_hidden_layers
        self.n_embd = config.hidden_size
        self.block_size = config.max_position_embeddings
        self.blocks = self.model.model.layers
        # Padding is pure bookkeeping: it is excluded from the pooled mean, and with right
        # padding and causal attention no real token ever attends to it. The filler only
        # has to be a valid embedding row, so 0 -- the same value the CrystaLLM backend
        # uses -- is fine.
        #
        # Deliberately NOT tokenizer.pad_token_id. That is <PAD> = 32004, one of five
        # tokens (<CLS> <SEP> <EOD> <MASK> <PAD>) that Megatron-LM's tokenizer adds by
        # default (llamat/Megatron-LLM/megatron/tokenizer/tokenizer.py:362) and the
        # Megatron -> HF conversion carried across. The embedding matrix was never
        # resized to match: it has 32,000 rows against the tokenizer's 32,005, so none of
        # those five can be embedded at all. They never appear in our text -- checked over
        # 3,000 real prompt+crystal-string sequences, the highest id produced is 29,999 --
        # so nothing here needs to defend against them; we simply do not use one.
        self.pad_id = 0
        self._prompt_text, self._prompt_ids = None, None   # constant prompt, tokenised once
        print(f"Loaded model: {self.n_layer} layers, {self.n_embd} dim, "
              f"block_size {self.block_size}, dtype {torch_dtype}")

    def encode(self, text: str):
        return self.tokenizer(text, add_special_tokens=True)["input_ids"]

    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def encode_pair(self, prompt: str, answer: str):
        """(ids, answer_start) for prompt+answer, tokenised as ONE string.

        The model must see the joined text, not two pieces glued together, so the whole
        thing is tokenised at once and the boundary is then located by checking that the
        result still starts with the prompt's own tokens. BPE can merge across a
        boundary ("output-" then "4"), which would shift the split by a token and
        silently pool one prompt position; the caller asserts the prefix matches.

        The prompt is constant across the corpus, so its ids are cached on first use.
        """
        if self._prompt_ids is None or prompt != self._prompt_text:
            self._prompt_text = prompt
            self._prompt_ids = self.encode(prompt) if prompt else self.encode("")
        # Deliberately NOT truncated here. Truncating inside the tokenizer would silently
        # drop atoms off the end of a crystal string and hand back an embedding of a
        # smaller cell; the caller checks the length against block_size and applies
        # --on-overflow instead.
        ids = self.encode(prompt + answer)
        start = len(self._prompt_ids)
        if ids[:start] != self._prompt_ids:
            # BPE merged across the boundary -- find the real split by re-encoding.
            start = _boundary(ids, self._prompt_ids)
        return ids, start

    def forward(self, input_ids):
        # No attention_mask: right-padding plus causal attention means real tokens never
        # see the pads, and mean_pool drops them from the average.
        self.model(input_ids)

    def generate_ids(self, prompt: str, max_new_tokens: int, temperature: float = 1.0,
                     top_k: int = None, top_p: float = None, use_cache: bool = True):
        """((1, T) token ids, prompt length)."""
        ids = self.encode(prompt)
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            y = self.model.generate(
                x,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k if top_k else 0,
                top_p=top_p if top_p is not None else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=use_cache,
            )
        return y, len(ids)

    def generate(self, prompt: str, max_new_tokens: int, **kw) -> str:
        """The CONTINUATION only, prompt stripped.

        The opposite of the CrystaLLM backend, and for the same reason: here the prompt
        is a 203-token instruction, not the head of the answer, so keeping it would put
        English prose in front of every crystal string and no parser would survive it.
        The slice is by token count, which is exact -- the prompt's own ids are a prefix
        of the generated sequence.
        """
        y, n_prompt = self.generate_ids(prompt, max_new_tokens, **kw)
        return self.decode(y[0][n_prompt:].tolist())


def _boundary(ids, prompt_ids) -> int:
    """First position where `ids` stops agreeing with `prompt_ids`.

    Only reached when BPE merges the last prompt token with the first answer token, which
    does not happen for the unconditional prompt (verified: the joined sequence starts
    with the prompt's own 203 ids). If it ever did, the merged token would be INCLUDED in
    the answer span, since mean_pool masks `pos >= start`. That is the right side to err
    on: a merged token carries answer content, so its hidden state varies with the
    structure, which is exactly what the pool is supposed to contain.
    """
    n = 0
    while n < len(prompt_ids) and n < len(ids) and ids[n] == prompt_ids[n]:
        n += 1
    return n


BACKENDS = {"crystallm": CrystaLLMBackend,
            "llamat2": LlamatBackend, "llamat2_cif": LlamatBackend}
assert set(BACKENDS) == set(MODELS), "every registered model needs a backend"
