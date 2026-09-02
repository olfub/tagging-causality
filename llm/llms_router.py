import asyncio, time
from contextlib import asynccontextmanager
from openai import OpenAI, AsyncOpenAI
from pathlib import Path
import json


class LLMInterface:

    def __init__(self, model_name, model_id, api_key=None, temperature=None, max_tokens=None):
        self.model_name = model_name
        self.model_id = model_id

    def query(self, messages, dry_run=False, **kwargs):
        if dry_run:
            print(messages)
            return None

        return self.do_query(messages, **kwargs)

    def batch_queries(self, batch_messages, dry_run=False, **kwargs):
        if dry_run:
            print(batch_messages)
            return None

        return self.do_queries(batch_messages, **kwargs)

    def do_query(self, messages, **kwargs) -> str:
        raise NotImplementedError()

    def do_queries(self, batch_messages, **kwargs) -> str:
        raise NotImplementedError()



class RateLimiter:
    def __init__(self, max_requests_per_minute, max_concurrent_requests):
        self.period = 60.0
        self.rate_limit = max_requests_per_minute
        self._max_concurrent_requests = max_concurrent_requests
        self.timestamps = []
        self._semaphore = None
        self._lock = None
        self._loop = None

    def _ensure_primitives(self):
        """Create/rebind sync primitives for the currently running loop."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self._max_concurrent_requests)
            self._lock = asyncio.Lock()
        return self._semaphore

    def get_max_concurrent_requests(self) -> int:
        return self._max_concurrent_requests

    async def _reserve_slot(self):
        async with self._lock:
            while True:
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < self.period]
                if len(self.timestamps) < self.rate_limit:
                    self.timestamps.append(now)
                    return
                sleep_time = self.period - (now - self.timestamps[0])
                await asyncio.sleep(max(sleep_time, 0.01))

    @asynccontextmanager
    async def slot(self):
        sem = self._ensure_primitives()
        async with sem:              # held for the whole request
            await self._reserve_slot()
            yield


class OpenRouterInterface(LLMInterface):
    def __init__(self, model_name, model_id, api_key=None, referer_url=None, app_title=None,
                 requests_per_minute=500, max_concurrent_requests=100,
                 base_url=None,
                 temperature=None, max_tokens=None,
                 extra_body=None):
        super().__init__(model_name, model_id, api_key)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.extra_body = extra_body

        if base_url is None:
            base_url = "https://openrouter.ai/api/v1"
        
        self._api_key = api_key
        self._base_url = base_url or "https://openrouter.ai/api/v1"
        self._async_client = None
        self._client_loop = None
        self.limiter = RateLimiter(requests_per_minute, max_concurrent_requests)

        # Headers for OpenRouter rankings/analytics
        #self.extra_headers = {}
        #if referer_url:
        #    self.extra_headers["HTTP-Referer"] = referer_url
        #if app_title:
        #    self.extra_headers["X-Title"] = app_title

    def _get_client(self):
        loop = asyncio.get_running_loop()
        if self._client_loop is not loop:
            self._async_client = AsyncOpenAI(
                api_key=self._api_key, base_url=self._base_url,
                max_retries=5
            )
            self._client_loop = loop
        return self._async_client

    def do_query(self, messages, temperature=0.0, max_tokens=4096):
        raise NotImplementedError("use async instead")
        #"""Standard synchronous query for a single request."""
        #completion = self.client.chat.completions.create(
        #    model=self.model_id,
        #    messages=messages,
        #    max_tokens=max_tokens,
        #    temperature=temperature,
        #    extra_headers=self.extra_headers
        #)
        #return completion.choices[0].message.content


    async def _process_single_async(self, messages, temperature, max_tokens):
        async with self.limiter.slot():
            try:
                completion = await self._get_client().chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=self.extra_body,
                )
                choice = completion.choices[0]
                add = ""
                #if getattr(choice, "finish_reason", "stop") != "stop":
                #    add = f"\n\n[[FINISHREASON]]{choice.finish_reason}"
                if getattr(choice, "finish_reason", None) != "stop" or choice.message.content is None:
                    return RuntimeError(f"Response finished unexpectedly. (finish_reason: {getattr(choice, 'finish_reason', None)}")
                return choice.message.content + add
            except Exception as e:
                return e

    def do_queries(self, batch_messages, temperature=None, max_tokens=None):
        """
        Takes a list of message lists (batch) and processes them in parallel 
        respecting the rate limit.
        """
        if temperature is None:
            temperature = self._temperature
        if max_tokens is None:
            max_tokens = self._max_tokens
        async def run_batch():
            try:
                tasks = [
                    self._process_single_async(msgs, temperature, max_tokens) 
                    for msgs in batch_messages
                ]
                return await asyncio.gather(*tasks)
            finally:
                if self._async_client is not None:
                    await self._async_client.close()
                    self._async_client = None
                    self._client_loop = None

        # Run the async loop
        return asyncio.run(run_batch())


llm_configs = {
    "openai--gpt-5.6-sol": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "openai/gpt-5.6-sol",
        "create_parameters": {
            #requests_per_minute=500, max_concurrent_requests=200,
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "openai--gpt-5.5": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "openai/gpt-5.5",
        "create_parameters": {
            #requests_per_minute=500, max_concurrent_requests=200,
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "openai--gpt-5.4": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "openai/gpt-5.4",
        "create_parameters": {
            #requests_per_minute=500, max_concurrent_requests=200,
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "openai--gpt-5.2": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "openai/gpt-5.2",
        "create_parameters": {
            #requests_per_minute=500, max_concurrent_requests=200,
            "temperature": 1.0,
            "max_tokens": 20000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "anthropic--claude-opus-5": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "anthropic/claude-opus-5",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "anthropic--claude-opus-4.8": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "anthropic/claude-opus-4.8",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "anthropic--claude-opus-4.7": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "anthropic/claude-opus-4.7",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "anthropic--claude-opus-4.6": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "anthropic/claude-opus-4.6",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "google--gemini-3-pro-preview": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "google/gemini-3-pro-preview",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 20000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "meta-llama--llama-3.3-70b-instruct": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 25000,
        }
    },
    "qwen--qwen3.8-max": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "qwen/qwen3.8-max",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "qwen--qwen3.7-max": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "qwen/qwen3.7-max",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "qwen--qwen3.6-max-preview": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "qwen/qwen3.6-max-preview",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 50000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "qwen--qwen3.5-397b-a17b": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "qwen/qwen3.5-397b-a17b",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 25000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "z-ai--glm-5": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "z-ai/glm-5",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 25000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
    "minimax--minimax-m2.5": {
        "keyname": "openrouter",
        "interface": OpenRouterInterface,
        "model_id": "minimax/minimax-m2.5",
        "create_parameters": {
            "temperature": 1.0,
            "max_tokens": 25000,
            "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
        }
    },
}


class LLMProvider:

    def __init__(self, key_dir=None):
        self._cache = {}
        self._key_dir: Path = key_dir

    def _create_interface(self, model_name) -> LLMInterface:
        llm_config = llm_configs[model_name]
        model_id = llm_config["model_id"]

        api_key = None
        if self._key_dir is not None:
            keyname = llm_config["keyname"]
            if keyname is not None:
                key_file = self._key_dir / keyname
                if key_file.exists():
                    api_key = key_file.read_text().strip()
                else:
                    raise FileNotFoundError(f"Key file {key_file} not found")

        create_parameters = llm_config.get("create_parameters", {})
        interface = llm_config["interface"](model_name, model_id, api_key, **create_parameters)
        return interface

    def get_interface(self, name) -> LLMInterface:
        interface = self._cache.get(name, None)
        if interface is not None:
            return interface

        interface = self._create_interface(name)
        self._cache[name] = interface
        return interface

