# ComfyUI VLM Prompt Enhancer

A single ComfyUI node that uses a local vision-language model to improve an image or video generation prompt.

## Remote OpenAI-compatible endpoint

When `VLM_PROMPT_ENHANCER_BASE_URL` and `VLM_PROMPT_ENHANCER_MODEL` are set,
the node sends prompt and reference media to that endpoint instead of loading a
Hugging Face model in ComfyUI. `VLM_PROMPT_ENHANCER_API_KEY` is optional, and
`VLM_PROMPT_ENHANCER_TIMEOUT` defaults to 600 seconds. Existing workflows keep
the same node inputs; the configured remote model overrides the local model and
quantization selectors.

The node downloads the selected model from Hugging Face on its first run. It caches files in `ComfyUI/models/vlm_prompt_enhancer`. Later runs use the local copy.

## Features

- Qwen 3.5 4B, 9B, and 27B presets
- Gemma 4 E2B, E4B, and 12B Instruct presets
- Any compatible Hugging Face model ID through `custom_model_id`
- Auto/BF16, FP16, 8-bit, and 4-bit NF4 loading
- Optional ComfyUI `IMAGE` batch and native `VIDEO` reference inputs
- System prompt, output length, temperature, top-p, seed, and video-frame controls
- One `STRING` output

## Install

Open a terminal in `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/cebeuq/comfyui-vlm-prompt-enhancer.git
cd comfyui-vlm-prompt-enhancer
pip install -r requirements.txt
```

For NVIDIA 4-bit and 8-bit loading:

```bash
pip install -r requirements-quantization.txt
```

Restart ComfyUI. Add **VLM Prompt Enhancer** from **text → prompting**.

## Use

1. Connect an optional image batch or native ComfyUI video.
2. Enter the prompt that you want to improve.
3. Select a model and precision.
4. Run the workflow.
5. Connect `enhanced_prompt` to a text encoder or another prompt node.

`reference_images` accepts a batch. This is also useful with video loaders that output frames as an `IMAGE` batch. `reference_video` accepts ComfyUI's native `VIDEO` type and samples frames uniformly.

Set temperature to `0` for deterministic greedy generation. Enable `unload_after_run` when VRAM is limited. The default keeps one model loaded for faster later runs. Selecting a different model or quantization unloads the old model.

## Model access and disk use

Model downloads are several gigabytes. Gemma checkpoints can require accepting Google's terms on Hugging Face. If a gated download fails, sign in with the Hugging Face CLI or set `HF_TOKEN` in the environment that starts ComfyUI.

Quantization applies while loading. Hugging Face still downloads the source checkpoint. The 4-bit and 8-bit choices use bitsandbytes and are intended primarily for NVIDIA GPUs. Use Auto/BF16 or FP16 on Apple silicon.

## Compatibility

The node uses the current Transformers multimodal chat-template API and `AutoModelForMultimodalLM`. It requires a recent ComfyUI version for the native `VIDEO` socket. The `IMAGE` input works on older versions.

## License

MIT
