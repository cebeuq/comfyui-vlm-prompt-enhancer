from __future__ import annotations

import gc
import os
from pathlib import Path

from .media import build_messages, tensor_to_pil_images, video_to_array
from .model_config import MODEL_IDS, QUANTIZATIONS, resolve_model_id


DEFAULT_SYSTEM_PROMPT = """You improve prompts for image and video generation.
Preserve the user's intent. Use the references to add accurate visual details.
Return only the improved prompt. Do not add a preface, notes, or quotation marks."""


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
    DESCRIPTION = "Enhance a prompt with a local Hugging Face vision-language model and optional image or video references."

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

        import torch

        model_id = resolve_model_id(model, custom_model_id)
        loaded_model, processor = self._load(model_id, quantization)

        images = tensor_to_pil_images(reference_images) if reference_images is not None else []
        video = video_to_array(reference_video, max_video_frames) if reference_video is not None else None
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
