
import time
from typing import Any, Dict, Optional

import openai
import tiktoken

from common.registry import registry


@registry.register_llm("gpt")
class OPENAI_GPT:
    def __init__(self,
                 engine="gpt-3.5-turbo-0631",
                 temperature=0,
                 max_tokens=200,
                 base_url=None,
                 api_key=None,
                 top_p=1,
                 stop=["\n"],
                 response_format=None,
                 seed=None,
                 top_logprobs=None,
                 retry_delays=60,
                 max_retry_iters=5,
                 context_length=4096,
                 system_message='',
                 timeout=None,
                 reasoning=None):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key
        self.top_p = top_p
        self.stop = stop
        self.response_format = response_format
        self.seed = seed
        self.top_logprobs = top_logprobs
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.context_length = context_length
        self.system_message = system_message
        self.timeout = timeout


        self.reasoning = reasoning
        self.last_usage: Dict[str, Any] = {}


        self.total_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "n_calls": 0,
        }
        self.client = self._build_client()


        try:
            self._encoding = tiktoken.encoding_for_model(self.engine)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def _build_client(self):

        api_key = self._api_key()
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return openai.OpenAI(**kwargs)

    def _api_key(self) -> str:
        if self.api_key:
            return self.api_key
        raise RuntimeError("Pass api_key in the runtime LLM configuration.")

    def _completion_kwargs(self, messages, *, logprobs: bool = False) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.engine,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.stop:
            kwargs["stop"] = self.stop
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if logprobs:
            kwargs["logprobs"] = True
            if self.top_logprobs is not None:
                kwargs["top_logprobs"] = self.top_logprobs
        if self.reasoning is not None:
            kwargs["extra_body"] = {"reasoning": self.reasoning}
        return kwargs

    def _accumulate_usage(self, usage: Dict[str, Any]) -> None:

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                self.total_usage[key] += value
        self.total_usage["n_calls"] += 1

    @staticmethod
    def _usage_dict(response) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    @staticmethod
    def _choice_text(choice) -> str:
        text = getattr(getattr(choice, "message", None), "content", None) or ""
        if not text.strip():
            raise RuntimeError("Empty content in OpenAI-compatible response")
        return text

    def llm_inference(self, messages):

        response = self.client.chat.completions.create(
            **self._completion_kwargs(messages)
        )


        text = self._choice_text(response.choices[0])
        self.last_usage = self._usage_dict(response)
        self._accumulate_usage(self.last_usage)
        return text

    def llm_inference_with_logprobs(self, messages):
        response = self.client.chat.completions.create(
            **self._completion_kwargs(messages, logprobs=True)
        )

        choice = response.choices[0]
        text = self._choice_text(choice)
        self.last_usage = self._usage_dict(response)
        self._accumulate_usage(self.last_usage)
        mean_lp: Optional[float] = None
        try:
            lp_obj = getattr(choice, "logprobs", None)
            content = getattr(lp_obj, "content", None) if lp_obj is not None else None
            if content:
                lps = [
                    float(t.logprob)
                    for t in content
                    if getattr(t, "logprob", None) is not None
                ]
                if lps:
                    mean_lp = sum(lps) / len(lps)
        except Exception:
            mean_lp = None
        return text, mean_lp

    def generate(self, system_message, prompt):
        messages = [
            {"role": "system", "content": system_message or self.system_message or ""},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(self.max_retry_iters):
            try:
                return True, self.llm_inference(messages)
            except Exception as e:
                print(f"[gpt] attempt {attempt + 1}/{self.max_retry_iters} failed: {e}")
                if attempt < self.max_retry_iters - 1:
                    time.sleep(self.retry_delays)
                else:
                    print("[gpt] giving up after multiple attempts.")
        return False, None

    def generate_raw(self, system_message, prompt):
        return self.generate(system_message, prompt)

    def generate_with_logprobs(self, system_message, prompt):
        messages = [
            {"role": "system", "content": system_message or self.system_message or ""},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(self.max_retry_iters):
            try:
                text, mean_lp = self.llm_inference_with_logprobs(messages)
                return True, text, mean_lp
            except Exception as e:
                print(f"[gpt] logprobs attempt {attempt + 1}/{self.max_retry_iters} failed: {e}")
                if attempt < self.max_retry_iters - 1:
                    time.sleep(self.retry_delays)
                else:
                    print("[gpt] giving up after multiple logprobs attempts.")
        return False, None, None

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):

        return sum(
            len(self._encoding.encode(v))
            for m in messages for v in m.values()
        ) + 3 * len(messages)

    @classmethod
    def from_config(cls, config):
        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        system_message = config.get("system_message", "You are a helpful assistant.")
        base_url = config.get("base_url", config.get("api_base", None))
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        response_format = config.get("response_format")
        seed = config.get("seed")
        top_logprobs = config.get("top_logprobs")
        retry_delays = config.get("retry_delays", 10)
        max_retry_iters = config.get("max_retry_iters", 5)
        context_length = config.get("context_length", 4096)
        timeout = config.get("timeout")
        reasoning = config.get("reasoning")
        return cls(engine=engine,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   base_url=base_url,
                   api_key=config.get("api_key"),
                   top_p=top_p,
                   response_format=response_format,
                   seed=seed,
                   top_logprobs=top_logprobs,
                   retry_delays=retry_delays,
                   max_retry_iters=max_retry_iters,
                   system_message=system_message,
                   context_length=context_length,
                   stop=stop,
                   timeout=timeout,
                   reasoning=reasoning)
