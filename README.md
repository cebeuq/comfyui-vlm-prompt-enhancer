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
- Qwen 3.8 27B presets, in BF16 and FP8
- Gemma 4 E2B, E4B, and 12B Instruct presets
- Any compatible Hugging Face model ID through `custom_model_id`
- Auto/BF16, FP16, 8-bit, and 4-bit NF4 loading
- Optional ComfyUI `IMAGE` batch and native `VIDEO` reference inputs
- A `mode` switch: `Default` keeps the free-text system prompt, `MiniMax H3` writes one for you
- MiniMax H3 task modes: T2V, I2V, FL2V, R2V, V2V, and the audio-synced I2VA, R2VA, and V2VA
- System prompt, output length, temperature, top-p, seed, and video-frame controls
- Two `STRING` outputs: the enhanced prompt, and the system prompt that was used

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

## MiniMax H3 mode

MiniMax H3 is an open-weights video model. It does not read a free-form
sentence. It reads a small set of named fields, and it binds a reference
picture to a subject or to a point in time through inline tags such as
`<Picture 1>`.

Set `mode` to `MiniMax H3` and the node stops using the `system_prompt`
widget. It builds the system prompt itself, from the task you choose and from
the references you connected, and it repairs the model's answer into the exact
field layout H3 expects.

### Tasks

| Task | Meaning | Pictures | What the picture fixes |
| --- | --- | --- | --- |
| `T2V` | Text to video | 0 | Nothing. The prompt carries everything. |
| `I2V` | Image to video | 1 | The exact first frame. |
| `FL2V` | First and last frame to video | 2 | The exact first frame and the exact last frame. |
| `R2V` | Reference to video | 1 to 9 | Subject identity only, not timing or framing. |
| `V2V` | Video to video | 0 to 3 | A reference video fixes motion and pacing. |
| `I2VA`, `R2VA`, `V2VA` | The same three, with a reference sound or voice track | as above | as above, plus lip and body sync. |

`h3_task` defaults to `Auto`. Auto reads the task from what you connected: no
reference is `T2V`, one picture is `I2V`, two pictures are `FL2V`, three or
more are `R2V`, a connected `reference_video` is `V2V`, and a connected
`reference_audio` selects the audio-synced variant.

### Picture order

H3 numbers pictures in the order it receives them. The node fills the slots in
this fixed order:

1. `first_frame` becomes `<Picture 1>`.
2. `last_frame` becomes the next picture.
3. `reference_images` fills the remaining slots, in batch order.

Wire `first_frame` and `last_frame` for `I2V` and `FL2V`. Wire a batch into
`reference_images` for `R2V`.

### Alignment line

H3 binds a picture to a moment with a fixed sentence at the top of the prompt.
The node writes that line itself, because a small model cannot be trusted to
copy a fixed string and a formatted number without drift. `I2V` gets the
zero-second binding. `FL2V` gets a two-mark binding built from
`h3_video_seconds`. `R2V` gets no binding line unless you switch on
`h3_anchor_first_reference`, which you should do only when the first reference
picture is also the opening frame.

### Length

`h3_video_seconds` sets the clip length. It drives the word budget, the
coverage checklist, and the second mark in the `FL2V` alignment line. H3 was
trained between about 2 and 15 seconds at 24 frames per second. In H3 mode the
node also raises `max_new_tokens` on its own when the budget needs more room.

### Why there are no examples in the built-in prompts

The built-in system prompts state rules. They contain no example scene, no
example subject, and no finished example prompt. A language model optimises
toward whatever it is shown. A worked example inside a system prompt leaks its
subject, its setting, and its wording into every later generation. Keeping the
prompt neutral keeps the output driven by your idea and your references.

`system_prompt_used` returns the exact text that was sent, so you can read what
the model was told.

Set temperature to `0` for deterministic greedy generation. Enable `unload_after_run` when VRAM is limited. The default keeps one model loaded for faster later runs. Selecting a different model or quantization unloads the old model.

## Model access and disk use

Model downloads are several gigabytes. Gemma checkpoints can require accepting Google's terms on Hugging Face. If a gated download fails, sign in with the Hugging Face CLI or set `HF_TOKEN` in the environment that starts ComfyUI.

Quantization applies while loading. Hugging Face still downloads the source checkpoint. The 4-bit and 8-bit choices use bitsandbytes and are intended primarily for NVIDIA GPUs. Use Auto/BF16 or FP16 on Apple silicon.

## Compatibility

The node uses the current Transformers multimodal chat-template API and `AutoModelForMultimodalLM`. It requires a recent ComfyUI version for the native `VIDEO` socket. The `IMAGE` input works on older versions.

## License

MIT
