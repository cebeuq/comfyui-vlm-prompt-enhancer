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
- MiniMax H3 task modes under MiniMax's own names: T2VA, I2VA, L2VA, FL2VA, and Ref2VA
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

MiniMax H3 is an open-weights video model. It always produces audio with the
video, which is why every mode name below ends in A. H3 does not read a
free-form sentence. It reads a small set of named fields, and it binds a
reference picture to a subject or to a point in time through inline tags such
as `<Picture 1>`.

MiniMax runs a hosted prompt-enrichment service, H3-Context-IR, that turns a
short idea into that field layout. It is not part of the open release. This
mode is the local stand-in for it.

Set `mode` to `MiniMax H3` and the node stops using the `system_prompt`
widget. It builds the system prompt itself, from the task you choose and from
the references you connected, writes the alignment line for you, and repairs
the model's answer into the exact field layout H3 expects.

### Tasks

| Task | Meaning | Pictures | What the picture fixes |
| --- | --- | --- | --- |
| `T2VA` | Text to video | 0 | Nothing. The prompt carries everything. |
| `I2VA` | Image to video | 1 | The exact first frame. |
| `L2VA` | Last frame to video | 1 | The exact last frame. The opening is invented. |
| `FL2VA` | First and last frame to video | 2 | Both ends of one continuous shot. |
| `Ref2VA` | Reference to video | 0 to 9 | Subject identity only, not timing or framing. Also takes reference video and reference audio. |

`h3_task` defaults to `Auto`. Auto reads the task from what you connected: no
reference is `T2VA`, one picture is `I2VA`, two pictures are `FL2VA`, and three
or more pictures, or any reference video or audio, is `Ref2VA`. Pick `L2VA`
by hand, because one picture cannot say by itself whether it is the first
frame or the last one.

The older widget values `T2V`, `I2V`, `FL2V`, `R2V`, `V2V`, `V2VA`, and `R2VA`
still load. They map onto the names above, so saved workflows keep working.

### Picture order

H3 numbers pictures in the order it receives them. The node fills the slots in
this fixed order:

1. `first_frame` becomes `<Picture 1>`.
2. `last_frame` becomes the next picture.
3. `reference_images` fills the remaining slots, in batch order.

Wire `first_frame` for `I2VA`, `first_frame` and `last_frame` for `FL2VA`, and
a batch into `reference_images` for `Ref2VA`. The node refuses a picture count
the task cannot use, rather than silently sending the wrong number.

### Alignment line

H3 binds a picture to a moment with a fixed sentence at the top of the prompt.
The node writes that line itself, because a small model cannot be trusted to
copy a fixed string and a formatted number without drift. MiniMax words the
line differently for each mode, and those differences are reproduced exactly:
`I2VA` gets the zero-second binding, `L2VA` and `FL2VA` get mark-based
bindings built from `h3_video_seconds`. `Ref2VA` gets no binding line unless
you switch on `h3_anchor_first_reference`, which you should do only when the
first reference picture is also the opening frame.

### Reference task label

`Ref2VA` prompts carry a bracketed label on the `summary` line that tells H3
what kind of reference job this is. `h3_ref_kind` defaults to `Auto`, which
derives the label from the connected references. A reference video that only
supplies camera work or rhythm stays `reference generation`; pick
`video editing` by hand when you actually mean it.

### Length

`h3_video_seconds` sets the clip length in whole seconds. The cloud endpoint
accepts 4 to 15, and the ComfyUI nodes snap the frame count to a 17k+5 grid at
24 frames per second. The value drives the word budget, the coverage
checklist, and the mark in the `L2VA` and `FL2VA` alignment lines. `Ref2VA`
carries MiniMax's longer 350 to 500 word budget. In H3 mode the node raises
`max_new_tokens` on its own when the budget needs more room.

### Why there are no examples in the built-in prompts

The built-in system prompts state rules. They contain no example scene, no
example subject, and no finished example prompt. A language model optimises
toward whatever it is shown. A worked example inside a system prompt leaks its
subject, its setting, and its wording into every later generation. Keeping the
prompt neutral keeps the output driven by your idea and your references.

`system_prompt_used` returns the exact text that was sent, so you can read what
the model was told.

## Vision and reasoning notes

Multi-image runs pass `add_vision_id` to the chat template, so the template
writes `Picture N:` before each image. Without it the model cannot tell which
picture is which, and every `<Picture N>` binding becomes a guess.

Each image is capped at about one million pixels, which is roughly a thousand
tokens. The shipped Qwen defaults allow far more, and nine reference pictures
at that setting fill a whole context before any text. Override it with
`VLM_PROMPT_ENHANCER_MAX_PIXELS` if you need finer detail.

Recent Qwen models think by default. The node asks for thinking to be turned
off, and it also strips a leaked `<think>` block from the answer, because an
OpenAI-compatible proxy does not always honour that request.

## Model access and disk use

Model downloads are several gigabytes. Gemma checkpoints can require accepting Google's terms on Hugging Face. If a gated download fails, sign in with the Hugging Face CLI or set `HF_TOKEN` in the environment that starts ComfyUI.

Quantization applies while loading. Hugging Face still downloads the source checkpoint. The 4-bit and 8-bit choices use bitsandbytes and are intended primarily for NVIDIA GPUs. Use Auto/BF16 or FP16 on Apple silicon.

## Compatibility

The node uses the current Transformers multimodal chat-template API and `AutoModelForMultimodalLM`. It requires a recent ComfyUI version for the native `VIDEO` socket. The `IMAGE` input works on older versions.

## License

MIT
