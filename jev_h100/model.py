from contextlib import nullcontext
import inspect

import torch
from torch import nn
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .common import LETTERS, view

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
           "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj"]


def legacy_qk_normalize(x):
    """Match mlx-lm 0.31.3: rms_norm(x, eps=1e-6) / sqrt(head_dim).

    Its effective sum-of-squares epsilon is head_dim*1e-6. Stock HF/FLA
    use 1e-6 on the sum instead. Keep the trained v1 convention explicit.
    The chunk kernel applies the additional query-only 1/sqrt(head_dim).
    """
    xf=x.float()
    return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)*(x.shape[-1]**-.5)).to(x.dtype)


def install_cpu_compatibility():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    if not hasattr(mod,"_jev_original_chunk"):
        mod._jev_original_chunk=inspect.unwrap(mod.torch_chunk_gated_delta_rule)
    if not hasattr(mod,"_jev_original_conv"):
        mod._jev_original_conv=inspect.unwrap(mod.causal_conv1d_fn)
    # Transformers' optional-extension dispatcher can select a CUDA-only
    # convolution even for CPU tensors when the extension is installed.
    mod.causal_conv1d_fn=mod._jev_original_conv
    original=mod._jev_original_chunk
    def chunk(q,k,v,g,beta,**kwargs):
        if kwargs.get("use_qk_l2norm_in_kernel",False):
            q,k=legacy_qk_normalize(q),legacy_qk_normalize(k)
            kwargs["use_qk_l2norm_in_kernel"]=False
        return original(q,k,v,g=g,beta=beta,**kwargs)
    mod.torch_chunk_gated_delta_rule=chunk


def install_cuda_kernels():
    """Explicit upstream differentiable kernels, never a silent slow fallback."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from causal_conv1d import causal_conv1d_fn
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    if not hasattr(mod,"_jev_original_chunk"):
        mod._jev_original_chunk=inspect.unwrap(mod.torch_chunk_gated_delta_rule)
    if not hasattr(mod,"_jev_original_conv"):
        mod._jev_original_conv=inspect.unwrap(mod.causal_conv1d_fn)
    allowed = set(inspect.signature(chunk_gated_delta_rule).parameters)
    if not {"q", "k", "v", "g", "beta", "use_qk_l2norm_in_kernel"} <= allowed:
        raise RuntimeError("unsupported FLA interface")

    def chunk(q, k, v, g, beta, **kwargs):
        kwargs = {key:value for key,value in kwargs.items() if key in allowed}
        if kwargs.get("use_qk_l2norm_in_kernel",False):
            q,k=legacy_qk_normalize(q),legacy_qk_normalize(k)
            kwargs["use_qk_l2norm_in_kernel"]=False
        return chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kwargs)

    def conv(x, weight, bias=None, activation=None, **kwargs):
        return causal_conv1d_fn(x, weight, bias=bias, activation=activation)

    mod.torch_chunk_gated_delta_rule = chunk
    mod.causal_conv1d_fn = conv
    return {"delta": "fla.ops.gated_delta_rule.chunk_gated_delta_rule", "conv": "causal_conv1d.causal_conv1d_fn"}


class DecisionModel(nn.Module):
    def __init__(self, lm, label_ids):
        super().__init__()
        self.lm = lm
        self.register_buffer("label_ids", torch.tensor(label_ids, dtype=torch.long), persistent=False)

    @property
    def base(self):
        return self.lm.get_base_model() if isinstance(self.lm, PeftModel) else self.lm

    def forward(self, input_ids, attention_mask, lengths, option_mask):
        h = self.base.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False).last_hidden_state
        last = h[torch.arange(h.shape[0], device=h.device), lengths-1]
        # Project only 26 rows. Do not allocate [batch,sequence,vocab] logits.
        rows = self.base.lm_head.weight[self.label_ids]
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = last.float() @ rows.float().T
        return logits.masked_fill(~option_mask, -1e9)


def decision_loss(logits, targets, mask):
    if targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("loss shape mismatch")
    if not torch.allclose(targets.sum(-1), torch.ones(targets.shape[0],device=targets.device),atol=1e-5):
        raise ValueError("target probability mass is not 1")
    if torch.any(targets[~mask] != 0):
        raise ValueError("target assigns mass to invalid options")
    logp = torch.log_softmax(logits.float().masked_fill(~mask,-1e9),dim=-1)
    return -(targets * logp).sum(-1).mean()


def load(path, device="cpu", adapter=None, train=False, gradient_checkpointing=False, kernels=True):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no CPU training fallback")
    if device == "cuda" and kernels:
        torch.backends.cuda.matmul.allow_tf32=False
        install_cuda_kernels()
    else:
        install_cpu_compatibility()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Preserve the exact v1 tokenizer; parity tests compare IDs, not just text.
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
    label_ids = []
    for c in LETTERS:
        ids = tokenizer.encode(" "+c, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"option {c} is not exactly one token")
        label_ids.append(ids[0])
    lm, loading = Qwen3_5ForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa", output_loading_info=True)
    if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
        raise RuntimeError(f"checkpoint loading mismatch: {loading}")
    lm.requires_grad_(False)
    if adapter:
        lm = PeftModel.from_pretrained(lm, adapter, is_trainable=train)
    elif train:
        lm = get_peft_model(lm, LoraConfig(r=32,lora_alpha=32,lora_dropout=0.,bias="none",
                                          target_modules=TARGETS,task_type=TaskType.CAUSAL_LM))
    if train:
        module_count=sum(1 for n,_ in lm.named_modules() if n.endswith("lora_A.default"))
        if module_count!=186:raise RuntimeError(f"unexpected Qwen3.5-2B LoRA coverage: {module_count}, expected 186")
    if train and gradient_checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        lm.enable_input_require_grads()
    model = DecisionModel(lm,label_ids).to(device)
    model.train(train)
    return model, tokenizer


def collate(rows, tokenizer, device="cpu", max_length=1024, rng=None, augment=False):
    views = [view(r,rng,augment) for r in rows]
    enc = tokenizer([r["prompt"] for r in views], add_special_tokens=False, padding=False)["input_ids"]
    lengths = [len(ids) for ids in enc]
    if max(lengths) > max_length:
        raise ValueError(f"input length {max(lengths)} exceeds frozen budget {max_length}; no truncation allowed")
    pad = tokenizer.pad_token_id
    if pad is None: pad = tokenizer.eos_token_id
    # Bound Triton shape variants and align GEMMs without truncating real tokens.
    padded_length=min(max_length,((max(lengths)+31)//32)*32)
    ids = torch.full((len(rows),padded_length),pad,dtype=torch.long)
    att = torch.zeros_like(ids)
    mask = torch.zeros((len(rows),26),dtype=torch.bool)
    targets = torch.zeros((len(rows),26),dtype=torch.float32)
    for i,(v,tokens) in enumerate(zip(views,enc)):
        ids[i,:len(tokens)]=torch.tensor(tokens)
        att[i,:len(tokens)]=1
        mask[i,:len(v["options"])]=True
        targets[i,:len(v["options"])]=torch.tensor(v["target"])
    return {"input_ids":ids.to(device),"attention_mask":att.to(device),
            "lengths":torch.tensor(lengths,device=device),"option_mask":mask.to(device)},targets.to(device),views


@torch.inference_mode()
def predict(model, tokenizer, rows, device="cpu", batch_size=16, max_length=1024):
    model.eval()
    results={}
    ordered=sorted(rows,key=lambda r:r.get("canonical_tokens",len(r["state"])))
    for off in range(0,len(ordered),batch_size):
        batch=ordered[off:off+batch_size]
        inputs,_,views=collate(batch,tokenizer,device,max_length)
        values=model(**inputs).float().cpu().numpy()
        for r,z in zip(views,values):
            results[r["uid"]]={k:r[k] for k in ["uid","task","kind","label","target","cluster_id","options"]}
            results[r["uid"]].update(logits=z[:len(r["options"])].tolist(),reference=r.get("reference","label"))
            for k in ["pair_id","workflow","parent_uid"]:
                if k in r:results[r["uid"]][k]=r[k]
    return [results[r["uid"]] for r in rows]
