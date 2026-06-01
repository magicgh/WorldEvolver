

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import uvicorn


class StopOnSubstring(StoppingCriteria):


    def __init__(self, tokenizer, stop_strings, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = [s for s in (stop_strings or []) if s]
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids, scores, **kwargs):
        if not self.stop_strings:
            return False
        new_tokens = input_ids[0][self.prompt_len:]
        if new_tokens.numel() == 0:
            return False
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    stop: Optional[Any] = None
    n: int = 1
    logprobs: bool = False
    top_logprobs: Optional[int] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    stop: Optional[Any] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = None


def _pydantic_dump(model_obj: BaseModel) -> Dict[str, Any]:

    dump = getattr(model_obj, "model_dump", None)
    if callable(dump):
        return dump()
    return model_obj.dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF OpenAI-compatible server")
    parser.add_argument("--model", default=os.environ.get("MODEL", "google/gemma-2-2b-it"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "30000")))
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "float16"),
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-context", type=int,
                        default=int(os.environ.get("CTX_LEN", "8192")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--attn", default=os.environ.get("ATTN_IMPL", "eager"),
                        help="HF attention backend: eager (V100-safe but slow), "
                             "sdpa (V100-fast), flash_attention_2 (Ampere+).")
    return parser.parse_args()


def build_app(model, tokenizer, model_name: str, max_context: int, device: str) -> FastAPI:
    app = FastAPI(title="hf-chat-shim", version="0.1")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "local"}],
        }

    def _token_logprob_payload(token_id: int, logprob: float,
                               top_ids: Optional[List[int]] = None,
                               top_lps: Optional[List[float]] = None) -> Dict[str, Any]:
        token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        top_payload: List[Dict[str, Any]] = []
        if top_ids is not None and top_lps is not None:
            for tid, lp in zip(top_ids, top_lps):
                top_text = tokenizer.decode([int(tid)], skip_special_tokens=False)
                top_payload.append({
                    "token": top_text,
                    "logprob": float(lp),
                    "bytes": list(top_text.encode("utf-8")),
                })
        return {
            "token": token_text,
            "logprob": float(logprob),
            "bytes": list(token_text.encode("utf-8")),
            "top_logprobs": top_payload,
        }

    def _collect_logprobs(generated: Any, new_ids, completion_tokens: int,
                          top_logprobs: Optional[int]) -> List[Dict[str, Any]]:
        scores = getattr(generated, "scores", None)
        if not scores:
            return []
        limit = min(int(completion_tokens), int(new_ids.shape[-1]), len(scores))
        top_k = max(0, int(top_logprobs or 0))
        payload: List[Dict[str, Any]] = []
        for i in range(limit):
            token_id = int(new_ids[i])
            lp_vec = torch.log_softmax(scores[i][0], dim=-1)
            token_lp = float(lp_vec[token_id].detach().cpu())
            if top_k > 0:
                vals, ids = torch.topk(lp_vec, k=min(top_k, lp_vec.shape[-1]))
                payload.append(_token_logprob_payload(
                    token_id,
                    token_lp,
                    [int(x) for x in ids.detach().cpu().tolist()],
                    [float(x) for x in vals.detach().cpu().tolist()],
                ))
            else:
                payload.append(_token_logprob_payload(token_id, token_lp))
        return payload

    def _generate(prompt_text: str, temperature: float, top_p: float,
                  max_tokens: int, stop: Optional[Any],
                  logprobs: bool = False,
                  top_logprobs: Optional[int] = None) -> Dict[str, Any]:
        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_context,
        ).to(device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        do_sample = bool(temperature and temperature > 0.0)
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)


        stop_list: List[str] = []
        if stop:
            stop_list = list(stop) if isinstance(stop, list) else [stop]
            stop_list = [s for s in stop_list if s]
        if stop_list:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopOnSubstring(tokenizer, stop_list, prompt_tokens),
            ])

        with torch.inference_mode():
            if logprobs:
                generated = model.generate(
                    **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                output_ids = generated.sequences
            else:
                generated = None
                output_ids = model.generate(**gen_kwargs)
        new_ids = output_ids[0][prompt_tokens:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)


        truncated_completion_tokens = int(new_ids.shape[-1])
        for s in stop_list:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                truncated_completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                break

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": truncated_completion_tokens,
            "logprobs": (
                _collect_logprobs(generated, new_ids, truncated_completion_tokens, top_logprobs)
                if logprobs else None
            ),
        }

    def _format_chat(messages: List[ChatMessage]) -> str:
        msgs = [_pydantic_dump(m) for m in messages]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chunks = []
            for m in msgs:
                chunks.append(f"{m['role']}: {m.get('content', '') or ''}")
            chunks.append("assistant:")
            return "\n".join(chunks)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        prompt_text = _format_chat(req.messages)
        result = _generate(prompt_text, req.temperature, req.top_p,
                           req.max_tokens, req.stop, req.logprobs, req.top_logprobs)
        choice: Dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }
        if req.logprobs:
            choice["logprobs"] = {"content": result["logprobs"] or []}
        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        })

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        result = _generate(req.prompt, req.temperature, req.top_p,
                           req.max_tokens, req.stop, req.logprobs, req.top_logprobs)
        choice: Dict[str, Any] = {
            "index": 0,
            "text": result["text"],
            "finish_reason": "stop",
        }
        if req.logprobs:
            choice["logprobs"] = {
                "tokens": [x["token"] for x in result["logprobs"] or []],
                "token_logprobs": [x["logprob"] for x in result["logprobs"] or []],
                "top_logprobs": [x["top_logprobs"] for x in result["logprobs"] or []],
            }
        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model or model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        })

    return app


def main() -> int:
    args = parse_args()
    print(f"[hf-server] loading {args.model} on {args.device} (dtype={args.dtype})", flush=True)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation=args.attn,
    )
    model.requires_grad_(False)
    print(f"[hf-server] ready; serving on {args.host}:{args.port}", flush=True)

    app = build_app(model, tokenizer, args.model, args.max_context, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
