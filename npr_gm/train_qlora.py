"""Local QLoRA pilot. No downloads or CUDA until explicit --run."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
from pathlib import Path
from .corpus import review_export, private_output
from .evaluate import load_cases

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local_input(path):
    path=Path(path).resolve()
    if "private_data" not in path.parts[:-1]:
        raise ValueError("Reviewed training inputs must be under private_data")
    return path

def prepare_rows(reviewed, split_map):
    reviewed=local_input(reviewed); split_map=local_input(split_map)
    rows=[json.loads(s) for s in reviewed.read_text(encoding="utf-8").splitlines() if s.strip()]
    mapping=json.loads(split_map.read_text(encoding="utf-8"))
    return review_export(rows,mapping,load_cases())

def encode_messages(tokenizer, messages, max_length):
    # Qwen chat template is authoritative. Never truncate a target silently.
    if [m["role"] for m in messages] != ["system","user","assistant"]:
        raise ValueError("Expected one reviewed turn")
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False,return_dict=False)
    if not isinstance(prefix,list) or not isinstance(full,list) or full[:len(prefix)]!=prefix:
        raise ValueError("Chat-template prefix mismatch; verify assistant masking")
    if len(full)>max_length:
        raise ValueError("Example exceeds max_length; shorten reviewed context, never truncate")
    if len(full)<=len(prefix):
        raise ValueError("No assistant target")
    return {"input_ids":full,"attention_mask":[1]*len(full),"labels":[-100]*len(prefix)+full[len(prefix):]}

def collate(rows,pad_id):
    size=max(len(r["input_ids"]) for r in rows)
    return {k:[r[k]+[fill]*(size-len(r[k])) for r in rows] for k,fill in
            (("input_ids",pad_id),("attention_mask",0),("labels",-100))}

def smoke_messages():
    # Tooling fixture only, not a game skill or evaluation item.
    return [{"messages":[{"role":"system","content":"Верни одно слово без пояснений."},
                        {"role":"user","content":"Напиши название месяца между мартом и маем."},
                        {"role":"assistant","content":"Апрель"}]}]

def validate_args(a):
    if not 128<=a.max_length<=2048 or not 1<=a.steps<=500 or not 1<=a.accum<=16:
        raise ValueError("Outside bounded pilot configuration")
    if a.smoke and (a.steps>3 or a.accum!=1):
        raise ValueError("Smoke limited to 3 optimizer steps, accumulation 1")
    if not a.smoke and (not a.reviewed or not a.split_map):
        raise ValueError("Reviewed data and explicit group split required")
    if a.smoke and (a.reviewed or a.split_map):
        raise ValueError("Do not mix fixture and corpus")
    if not re.fullmatch("[a-f0-9]{40}",a.revision):
        raise ValueError("Pin official HF commit, not main")
    if a.model not in {"Qwen/Qwen3.5-4B","Qwen/Qwen3.5-2B"}:
        raise ValueError("Unsupported base")
    if not math.isfinite(a.lr) or not 0<a.lr<=0.0002:
        raise ValueError("Learning rate outside pilot bound")

def run(a,rows):
    # Disable telemetry and external reporting. Download only official pinned model.
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["WANDB_DISABLED"]="true"
    import torch
    import bitsandbytes as bnb
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU training disabled")
    free,total=torch.cuda.mem_get_info()
    if free < 4*1024**3:
        raise RuntimeError("Less than 4 GiB free VRAM; close other GPU apps or try a smaller model")
    torch.manual_seed(a.seed);random.seed(a.seed)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision,trust_remote_code=False)
    encoded=[encode_messages(tok,x["messages"],a.max_length) for x in rows]
    if tok.pad_token_id is None:
        raise ValueError("Tokenizer has no pad token; inspect official config")
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    # Official multimodal checkpoint, text-only training input; no vision targets.
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,revision=a.revision,
        quantization_config=config,device_map={"":0},dtype=dtype,trust_remote_code=False)
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False})
    # Explicit names restrict training to text attention projections.
    targets=[name for name,module in model.named_modules()
             if "language_model" in name and name.rsplit(".",1)[-1] in {"q_proj","k_proj","v_proj","o_proj"}
             and isinstance(module,bnb.nn.Linear4bit)]
    if not targets:
        raise RuntimeError("No expected text projection layers; inspect model version")
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
        bias="none",target_modules=targets,task_type="CAUSAL_LM"))
    params=[p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError("No trainable parameters")
    optim=bnb.optim.PagedAdamW8bit(params,lr=a.lr)
    scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    model.train()
    out=private_output(a.output/"receipt.json").parent
    if (out/"receipt.json").exists() or (out/"adapter").exists(): raise FileExistsError("Use a new run directory")
    torch.cuda.reset_peak_memory_stats()
    losses=[]
    rng=random.Random(a.seed)
    for step in range(a.steps):
        optim.zero_grad(set_to_none=True)
        total_loss=0.0
        for _ in range(a.accum):
            row=encoded[rng.randrange(len(encoded))]
            batch={k:torch.tensor(v,device="cuda",dtype=torch.long) for k,v in collate([row],tok.pad_token_id).items()}
            with torch.autocast("cuda",dtype=dtype):
                loss=model(**batch).loss/a.accum
            if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
            scaler.scale(loss).backward()
            total_loss+=float(loss.detach().item())
        scaler.unscale_(optim)
        grad=torch.nn.utils.clip_grad_norm_(params,1.0)
        if not torch.isfinite(grad): raise RuntimeError("Non-finite gradient")
        scaler.step(optim);scaler.update()
        losses.append(total_loss)
        print(json.dumps({"step":step+1,"loss":total_loss,"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()}),flush=True)
    model.save_pretrained(out/"adapter",safe_serialization=True)
    tok.save_pretrained(out/"adapter")
    receipt={"status":"smoke_only_not_for_deployment" if a.smoke else "pilot_trained_not_evaluated",
        "model":a.model,"revision":a.revision,"steps":a.steps,"accum":a.accum,"seed":a.seed,"lr":a.lr,
        "max_length":a.max_length,"examples":len(encoded),"losses":losses,"dtype":str(dtype),
        "gpu":torch.cuda.get_device_name(),"vram_total":total,"vram_free_start":free,
        "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
        "trainable_parameters":sum(p.numel() for p in params),"target_modules":targets,
        "input_sha256":None if a.smoke else sha(a.reviewed),"split_sha256":None if a.smoke else sha(a.split_map),
        "packages":{k:importlib.metadata.version(k) for k in ("torch","transformers","peft","bitsandbytes","accelerate")},
        "adapter_sha256":sha(out/"adapter"/"adapter_model.safetensors")}
    with (out/"receipt.json").open("x",encoding="utf-8") as f: json.dump(receipt,f,indent=2,ensure_ascii=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="Qwen/Qwen3.5-4B")
    p.add_argument("--revision",required=True);p.add_argument("--reviewed",type=Path);p.add_argument("--split-map",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--smoke",action="store_true");p.add_argument("--run",action="store_true")
    p.add_argument("--max-length",type=int,default=512);p.add_argument("--steps",type=int,default=3);p.add_argument("--accum",type=int,default=1)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=42)
    a=p.parse_args();validate_args(a)
    rows=smoke_messages() if a.smoke else prepare_rows(a.reviewed,a.split_map)
    if "private_data" not in a.output.resolve().parts: raise ValueError("Output must remain private")
    print(json.dumps({"mode":"smoke" if a.smoke else "reviewed_pilot","examples":len(rows),"run":a.run,"steps":a.steps,"max_length":a.max_length}))
    if a.run: run(a,rows)
if __name__=="__main__":main()
