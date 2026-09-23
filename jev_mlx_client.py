"""Native Apple Silicon inference for HF and native MLX exports; no generation.

Tested with mlx-lm 0.31.3. The runtime model-type override keeps the same weights
usable by HF/CUDA; it does not rewrite the exported config or checkpoint.
"""
import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm import load

LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER="You are a decision function. Read the state, then answer the question by choosing exactly one option."


class NativeRMSNorm(nn.Module):
    """HF Qwen norm convention: FP32 gain multiplication, then input dtype."""
    def __init__(self,gain,eps):
        super().__init__();self.weight=gain;self.eps=eps
    def __call__(self,x):
        return mx.fast.rms_norm(x.astype(mx.float32),self.weight,self.eps).astype(x.dtype)


class Client:
    def __init__(self,model_path,calibration=None):
        self.model,self.tok=load(model_path,model_config={"model_type":"qwen3_5"},tokenizer_config={"fix_mistral_regex":False})
        cfg=json.loads((Path(model_path)/"config.json").read_text())
        native_export=cfg.get("jev_native_mlx_norm_gain") is True
        if cfg.get("model_type")=="qwen3_5_text" or native_export:
            suffixes=(".input_layernorm.weight",".post_attention_layernorm.weight","model.norm.weight",".q_norm.weight",".k_norm.weight")
            modules=[]
            for file in Path(model_path).glob("model*.safetensors"):
                raw=mx.load(str(file))
                for key,w in raw.items():
                    if key.endswith(suffixes):
                        name=key.removesuffix(".weight")
                        if not native_export:name="language_model."+name
                        gain=w.astype(mx.float32) if native_export else w.astype(mx.float32)+1.
                        modules.append((name,NativeRMSNorm(gain,cfg.get("rms_norm_eps",1e-6))))
                del raw
            self.model.update_modules(tree_unflatten(modules))
        self.model.eval()
        ids=[]
        for c in LETTERS:
            encoded=self.tok.encode(" "+c,add_special_tokens=False)
            if len(encoded)!=1:raise ValueError("option label must be one token")
            ids.append(encoded[0])
        self.label_ids=mx.array(ids)
        self.temperature=1.
        default=Path(model_path)/"calibration.mlx.json"
        if not default.exists():default=Path(model_path)/"calibration.json"
        path=Path(calibration) if calibration else default
        if path.exists():self.temperature=float(json.loads(path.read_text())["temperature"])
        if not math.isfinite(self.temperature) or self.temperature<=0:raise ValueError("invalid temperature")

    def decide(self,state,question,options):
        if not 2<=len(options)<=26 or len(set(options))!=len(options):raise ValueError("need 2–26 unique options")
        lines="\n".join(f"{LETTERS[i]}. {o}" for i,o in enumerate(options))
        prompt=f"{HEADER}\n\n[State]\n{state}\n\n[Question]\n{question}\n\n[Options]\n{lines}\n\nAnswer:"
        logits=mx.array(self.prompt_logits(prompt,len(options)))
        probs=mx.softmax(logits/self.temperature).tolist()
        return {"choice":options[max(range(len(probs)),key=probs.__getitem__)],"probabilities":dict(zip(options,probs))}

    def prompt_logits(self,prompt,n_options):
        if not 2<=n_options<=26:raise ValueError("need 2–26 options")
        ids=self.tok.encode(prompt,add_special_tokens=False)
        if len(ids)>1024:raise ValueError("input exceeds the validated 1024-token budget")
        inner=self.model.language_model.model
        last=inner(mx.array([ids]))[0,-1]
        weights=inner.embed_tokens.weight[self.label_ids[:n_options]]
        return (weights.astype(mx.float32)@last.astype(mx.float32)).tolist()

    def decide_bool(self,state,proposition):
        return self.decide(state,proposition,["yes","no"])["probabilities"]["yes"]

    def decide_score(self,state,question,levels):
        result=self.decide(state,question,levels)
        result["expected_level"]=sum(i*result["probabilities"][level] for i,level in enumerate(levels))
        return result

    def prompt_logits_batch(self,items):
        encoded=[self.tok.encode(x['prompt'],add_special_tokens=False) for x in items]
        lengths=[len(x) for x in encoded]
        if not lengths or max(lengths)>1024:raise ValueError('invalid context length')
        import numpy as np
        ids=np.zeros((len(items),max(lengths)),dtype=np.int32)
        for i,tokens in enumerate(encoded):ids[i,:len(tokens)]=tokens
        inner=self.model.language_model.model
        h=inner(mx.array(ids))
        last=h[mx.arange(len(items)),mx.array(lengths)-1].astype(mx.float32)
        weights=inner.embed_tokens.weight[self.label_ids].astype(mx.float32)
        values=(last@weights.T).tolist()
        return [z[:item['n_options']] for z,item in zip(values,items)]


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--calibration")
    p.add_argument("--state",required=True);p.add_argument("--question",required=True);p.add_argument("--options",nargs="+",required=True)
    a=p.parse_args()
    print(json.dumps(Client(a.model,a.calibration).decide(a.state,a.question,a.options),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
