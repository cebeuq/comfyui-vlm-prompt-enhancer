from __future__ import annotations

import base64
import gc
import json
import os
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .media import build_messages, tensor_to_pil_images, video_to_array
from .model_config import MODEL_IDS, QUANTIZATIONS, resolve_model_id


DEFAULT_SYSTEM_PROMPT = """You improve prompts for image and video generation.
Preserve the user's intent. Use the references to add accurate visual details.
Return only the improved prompt. Do not add a preface, notes, or quotation marks."""


def _pil_data_url(image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _remote_messages(system_prompt: str, prompt: str, images, video) -> list[dict]:
    from PIL import Image

    content = []
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": _pil_data_url(image)}})
    if video is not None:
        for frame in video:
            content.append(
                {"type": "image_url", "image_url": {"url": _pil_data_url(Image.fromarray(frame))}}
            )
    content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": content})
    return messages


def _enhance_remote(
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    prompt: str,
    images,
    video,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    payload = {
        "model": model,
        "messages": _remote_messages(system_prompt, prompt, images, video),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = float(os.environ.get("VLM_PROMPT_ENHANCER_TIMEOUT", "600"))
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote VLM request failed with HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Remote VLM endpoint is unavailable: {error.reason}") from error

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Remote VLM returned an invalid response: {result}") from error
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    content = str(content).strip()
    if not content:
        raise RuntimeError("Remote VLM returned an empty prompt")
    return content


def _model_cache_directory() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.models_dir)
    except ImportError:
        root = Path(os.environ.get("COMFYUI_MODEL_PATH", Path.cwd() / "models"))
    path = root / "vlm_prompt_enhancer"
    path.mkdir(parents=True, exist_ok=True)
    return path


class VLMPromptEnhancer:
    _loaded_key = None
    _model = None
    _processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt to improve"}),
                "model": (list(MODEL_IDS), {"default": "Qwen 3.5 4B"}),
                "quantization": (list(QUANTIZATIONS), {"default": "Auto / BF16"}),
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_video_frames": ("INT", {"default": 12, "min": 2, "max": 64}),
                "unload_after_run": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "reference_video": ("VIDEO",),
                "custom_model_id": ("STRING", {"default": "", "placeholder": "Optional: owner/model-name"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    OUTPUT_TOOLTIPS = ("The enhanced prompt, ready for a text encoder or prompt node.",)
    FUNCTION = "enhance"
    CATEGORY = "text/prompting"
    DESCRIPTION = "Enhance a prompt with the configured remote VLM or a local Hugging Face model."

    @classmethod
    def _unload(cls):
        cls._model = None
        cls._processor = None
        cls._loaded_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError):
            pass

    @classmethod
    def _load(cls, model_id: str, quantization: str):
        key = (model_id, quantization)
        if cls._loaded_key == key and cls._model is not None:
            return cls._model, cls._processor

        cls._unload()
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "Missing dependencies. Run: pip install -r custom_nodes/comfyui-vlm-prompt-enhancer/requirements.txt"
            ) from error

        load_kwargs = {
            "cache_dir": str(_model_cache_directory()),
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if quantization == "FP16":
            load_kwargs["dtype"] = torch.float16
        elif quantization == "8-bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4-bit NF4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            load_kwargs["dtype"] = "auto"

        try:
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(_model_cache_directory()))
            model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
        except ImportError as error:
            if quantization in {"8-bit", "4-bit NF4"}:
                raise RuntimeError(
                    "4-bit and 8-bit modes need bitsandbytes. Install the optional quantization requirements."
                ) from error
            raise

        model.eval()
        cls._loaded_key = key
        cls._model = model
        cls._processor = processor
        return model, processor

    def enhance(
        self,
        prompt,
        model,
        quantization,
        system_prompt,
        max_new_tokens,
        temperature,
        top_p,
        seed,
        max_video_frames,
        unload_after_run,
        reference_images=None,
        reference_video=None,
        custom_model_id="",
    ):
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        images = tensor_to_pil_images(reference_images) if reference_images is not None else []
        video = video_to_array(reference_video, max_video_frames) if reference_video is not None else None
        remote_base_url = os.environ.get("VLM_PROMPT_ENHANCER_BASE_URL", "").strip()
        if remote_base_url:
            remote_model = os.environ.get("VLM_PROMPT_ENHANCER_MODEL", "").strip()
            if not remote_model:
                raise RuntimeError(
                    "VLM_PROMPT_ENHANCER_MODEL is required when VLM_PROMPT_ENHANCER_BASE_URL is set"
                )
            result = _enhance_remote(
                remote_base_url,
                remote_model,
                (os.environ.get("VLM_PROMPT_ENHANCER_API_KEY") or os.environ.get("LLAMA_API_KEY") or "").strip(),
                system_prompt,
                prompt.strip(),
                images,
                video,
                max_new_tokens,
                temperature,
                top_p,
                seed,
            )
            return (result,)

        import torch

        model_id = resolve_model_id(model, custom_model_id)
        loaded_model, processor = self._load(model_id, quantization)

        messages = build_messages(system_prompt, prompt.strip(), images, video)
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            inputs = processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
        except TypeError:
            # Some compatible custom processors do not expose a thinking switch.
            inputs = processor.apply_chat_template(messages, **template_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        target_device = next(loaded_model.parameters()).device
        inputs = inputs.to(target_device)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        sampling = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sampling,
        }
        if sampling:
            generation_kwargs.update(temperature=temperature, top_p=top_p)

        try:
            with torch.inference_mode():
                output = loaded_model.generate(**inputs, **generation_kwargs)
            result = processor.decode(output[0][input_length:], skip_special_tokens=True).strip()
        finally:
            if unload_after_run:
                self._unload()

        return (result,)


NODE_CLASS_MAPPINGS = {"VLMPromptEnhancer": VLMPromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"VLMPromptEnhancer": "VLM Prompt Enhancer"}
