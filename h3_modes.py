"""MiniMax H3 prompt modes for the VLM Prompt Enhancer node.

This module holds everything that is specific to the MiniMax H3 video model:

* the list of generation modes the model accepts,
* automatic mode detection from the connected reference inputs,
* the system prompt that is built for each mode,
* the alignment line that binds a reference picture to a point in time,
* a post-processor that repairs the field layout H3 expects.

The system prompts contain rules only. They contain no example scene, no
example subject, and no example finished prompt. A language model copies what
it is shown, so showing it a finished prompt biases every later generation
toward that content.
"""

from __future__ import annotations

# The prompt encoder of MiniMax H3 truncates long prompts. Keep the final
# string under this many characters.
H3_MAX_CHARS = 7000

MODE_DEFAULT = "Default"
MODE_H3 = "MiniMax H3"
MODES = (MODE_DEFAULT, MODE_H3)

# Task codes, in the order they are offered in the node.
H3_AUTO = "Auto"
H3_TASKS = ("T2V", "I2V", "FL2V", "R2V", "V2V", "I2VA", "R2VA", "V2VA")
H3_TASK_OPTIONS = (H3_AUTO,) + H3_TASKS

H3_TASK_LABELS = {
    "T2V": "Text to video. No reference media.",
    "I2V": "Image to video. One picture is the exact first frame.",
    "FL2V": "First and last frame to video. Two pictures bound the shot.",
    "R2V": "Reference to video. Pictures carry identity, not timing.",
    "V2V": "Video to video. A reference video carries motion and timing.",
    "I2VA": "Image to video with a reference sound or voice track.",
    "R2VA": "Reference to video with a reference sound or voice track.",
    "V2VA": "Video to video with a reference sound or voice track.",
}

# MiniMax H3 reads a small set of named fields. Reference tasks carry the
# longer set, because the model must be told which subject each picture fixes
# before the timeline starts.
_SIMPLE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
_REFERENCE_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


def field_schema(task: str) -> tuple[str, ...]:
    """Return the field names, in order, that H3 expects for a task."""
    if task in {"R2V", "R2VA"}:
        return _REFERENCE_FIELDS
    return _SIMPLE_FIELDS


def _body_field(schema: tuple[str, ...]) -> str:
    """Return the field that holds the timeline."""
    return "detailed_description" if "detailed_description" in schema else "integrated_multimodal_description"


# Tasks that consume the reference video slot.
_VIDEO_TASKS = frozenset({"V2V", "V2VA"})
# Tasks that consume a reference audio track.
_AUDIO_TASKS = frozenset({"I2VA", "R2VA", "V2VA"})

# The generation grid of H3 is 24 frames per second, and the trained clip
# length runs from about 2 seconds to about 15 seconds.
H3_FPS = 24
H3_MIN_SECONDS = 2.0
H3_MAX_SECONDS = 15.0


def detect_task(image_count: int, has_video: bool, has_audio: bool) -> str:
    """Choose a task code from the reference media that is connected."""
    if has_video and has_audio:
        return "V2VA"
    if has_video:
        return "V2V"
    if image_count >= 3:
        return "R2VA" if has_audio else "R2V"
    if image_count == 2:
        return "FL2V"
    if image_count == 1:
        return "I2VA" if has_audio else "I2V"
    return "T2V"


def resolve_task(selection: str, image_count: int, has_video: bool, has_audio: bool) -> str:
    """Return the task code to use for this run."""
    choice = (selection or H3_AUTO).strip()
    if choice and choice != H3_AUTO:
        if choice not in H3_TASKS:
            raise ValueError(f"Unknown MiniMax H3 task: {selection}")
        return choice
    return detect_task(image_count, has_video, has_audio)


def expected_image_count(task: str) -> tuple[int, int]:
    """Return the smallest and largest useful picture count for a task."""
    if task in {"T2V", "V2V", "V2VA"}:
        return (0, 3)
    if task in {"I2V", "I2VA"}:
        return (1, 1)
    if task == "FL2V":
        return (2, 2)
    return (1, 9)


def snap_seconds(seconds: float) -> float:
    """Clamp a requested duration into the range H3 was trained on."""
    value = float(seconds)
    if value < H3_MIN_SECONDS:
        return H3_MIN_SECONDS
    if value > H3_MAX_SECONDS:
        return H3_MAX_SECONDS
    return value


def alignment_line(task: str, image_count: int, seconds: float, anchor_first_reference: bool) -> str:
    """Build the reference-to-timeline binding line that H3 reads first.

    The node writes this line itself. A small model cannot be trusted to copy
    a fixed string and a formatted number without drift.
    """
    if image_count <= 0:
        return ""
    if task in {"I2V", "I2VA"}:
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if task == "FL2V" and image_count >= 2:
        end = "%.2f" % (int(round(snap_seconds(seconds) * 10000)) // 100 / 100.0)
        return (
            "How the reference pictures align with the target video — "
            "<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of the target video; "
            f"<Picture 2> (from [Shot 1]) aligns with the {end}-second mark of the target video."
        )
    if anchor_first_reference:
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    return ""


# --------------------------------------------------------------------------
# System prompt assembly
# --------------------------------------------------------------------------

_ROLE = """You write one prompt for the MiniMax H3 video model. You do not talk to the user.
The user gives you a short idea. The idea may be vague, misspelled, or only a few words.
You turn that idea into one finished H3 prompt.
You never ask a question. You never refuse. You never explain yourself."""

_FIELD_HELP = {
    "subject_definitions": "<name each subject once, and bind it to its picture tag>",
    "summary": "<one sentence that says what the video shows>",
    "retention_analysis": "<what stays identical to the references: face, hair, clothing, object shape>",
    "detailed_description": "[Shot 1] <the timeline>",
    "integrated_multimodal_description": "[Shot 1] <the timeline>",
    "overall_soundscape": "<the real sounds of that place>",
    "non_diegetic_music": "<the score the audience hears, or N/A>",
}


def _output_contract(schema: tuple[str, ...]) -> str:
    shape = "\n\n".join(f"{name}: {_FIELD_HELP[name]}" for name in schema)
    last = schema[-1]
    count = len(schema)
    return f"""OUTPUT SHAPE.
Your whole answer is these {len(schema)} fields, in this order, and nothing else:

{shape}

Put one empty line between the fields.
Never put a line break inside a field.
Never write a heading, a title, a bullet, a number list, a code fence, or quotation marks around a field.
Never write a greeting, a preface, a note, or a closing remark. Write nothing outside the fields.
Never repeat these instructions back.
Your answer ends at the end of the {last} line.
Write in English only. Never write Chinese characters.

FIELD NAMES.
Every field begins with its exact name, then a colon, then a space. Type the name even when the field is short.
Your answer contains all {count} field names, each one time, in the order above.
Never merge two fields into one. Never let the content of one field run on at the end of another field.
Never leave a field out. A field with nothing to say still gets its name, followed by N/A."""

_VIEWER_RULES = """WHAT THE VIEWER SEES.
Write only what a viewer can see or hear. Never write a smell, a taste, an emotion, or a thought.
The viewer watches a moving video. The words picture, image, photo, photograph, frame, reference, left, right, side by side, split screen and collage never appear inside any field.
Some words in the idea name the camera, not a thing in the video. Words such as drone, aerial, bird's eye, crane, helicopter, close-up, wide shot, tracking shot and POV describe how the shot is taken. The viewer never sees the camera or the operator. Turn such a word into a camera movement, and never put the device in the scene.
Never open the description by repeating the idea sentence."""

_STYLE_RULE = """STYLE SENTENCE.
Begin the description with one short style sentence. Name the medium and the overall light.
Name the medium with one of these exact terms and no other: live-action cinematic, documentary, 3D render, anime, illustration, stop motion, archival footage.
Read the medium off the reference media. A camera photograph is live-action cinematic. A hand drawing is anime or illustration. A computer render is 3D render.
When there is no reference media, pick the medium that fits the idea.
After the style sentence, name the subject. After the subject, write what happens."""

_SENTENCE_RULE = """SENTENCES.
Every sentence ends with a full stop. Never join two statements with a comma.
Use plain, direct language. Do not use poetic or abstract words.
Describe one thing at a time."""

_CAMERA_RULE = """CAMERA.
Write camera movement as something the camera does, inside a full sentence. Capitalise that sentence like any other.
Never write camera words as a tag list at the end.
Use at most two camera movements in the whole description.
Use only these movements, and copy the wording exactly: zooms in, zooms out, pushes in, pulls out, pans left, pans right, tilts up, tilts down, trucks left, trucks right, tracks forward, tracks backward, holds still.
Every camera movement ends with an amplitude phrase and then a speed phrase, in this exact wording: "with small amplitude at slow speed", "with small amplitude at fast speed", "with large amplitude at slow speed", or "with large amplitude at fast speed".
A calm scene takes small amplitude and slow speed. Only a genuinely energetic scene takes large amplitude or fast speed.
These words are banned everywhere: slightly, subtly, gently, a little, gradually, somewhat."""

_MOTION_RULE = """MOTION.
Something must move in every second of the video. A still description makes a still video.
Write motion as one continuous, physically possible chain. Each action causes the next one.
Never write a cut, a transition, a montage, a second scene, or a second shot. The whole answer holds one [Shot 1] and nothing else.
Never write a speed change such as slow motion or time lapse unless the idea asks for it."""

_TEXT_RULE = """ON-SCREEN TEXT.
Any words that must appear on screen, on a sign or a title, go inside straight double quotation marks. Write nothing else in quotation marks."""

_DIALOGUE_RULE = """SPEECH.
Somebody speaks only when the idea hands you words to say, or clearly asks for speech. When the idea gives you no line, write no speech at all.
When somebody does speak, the spoken words appear one time only, and they appear inside this wrapper: <d>[English] The words.</d>
Copy the square brackets and the tag exactly. The words start with a capital letter and end with a full stop, a question mark, or an exclamation mark.
Build the whole spoken passage in this shape, and replace only the capitalised parts:
HIS OR HER jaw and lips move clearly through every word, and the SPEAKER with a VOICE QUALITY voice (S1) says: <d>[English] THE WORDS.</d> He closes his lips and ONE ACTION.
Use (S1) for the first speaker and (S2) for a second speaker.
The capitals in that shape only mark the parts you replace. Write the finished sentence in ordinary sentence case, and never copy a capitalised placeholder into your answer.
Never summarise, hint at, or repeat the spoken words anywhere else."""

_SOUND_RULE = """SOUND FIELDS.
overall_soundscape lists three to five real sounds of the place, and nothing else.
No person makes a vocal sound in overall_soundscape. These words are banned from that field: voice, voices, talking, talk, chatter, murmur, mutter, conversation, speech, speaking, whisper, whispering, shouting, singing, lyrics, crowd noise.
A crowded place is still silent of people in this field. Name objects instead of people, such as cups, doors, machines, footsteps, or traffic.
Moving air, cloth, and machines never whisper or murmur. They hiss, rush, hum, rustle, creak, or sigh.
Never ask for silence or for clean audio. A room always has room tone.
non_diegetic_music names instruments, tempo, and volume. Write "N/A" when the scene should carry no score."""


def _slot_block(task: str, image_count: int, video_count: int, has_audio: bool) -> str:
    """Explain what each attached reference means for this task."""
    lines = ["REFERENCE MEDIA."]

    if image_count == 0 and video_count == 0:
        lines.append("No reference media is attached. You invent the whole scene from the idea.")
        lines.append(
            "Because nothing is fixed for you, describe the subject, the clothing, the surface "
            "materials, the setting, and the light in full. Leave nothing to chance."
        )
        return "\n".join(lines)

    if image_count == 1:
        lines.append("One picture is attached. It is <Picture 1>.")
    elif image_count > 1:
        names = ", ".join(f"<Picture {index + 1}>" for index in range(image_count))
        lines.append(
            f"{image_count} pictures are attached, in order. They are {names}. "
            "The order you are shown them in is the order of those numbers."
        )
    else:
        lines.append("No picture is attached.")

    if task in {"I2V", "I2VA"}:
        lines.append("<Picture 1> is the exact first frame of the video. The video starts inside it.")
        lines.append(
            "Describe the same person or object, the same clothing, the same place, and the same "
            "light that <Picture 1> shows. Then describe what happens next."
        )
        lines.append("Never invent a different scene. Never change the clothing. Never add a person who is not in <Picture 1>.")
        lines.append(
            "Appearance is already fixed by the picture, so spend your words on movement, on physics, "
            "and on what changes over time."
        )
        lines.append(
            "The binding line above your answer already ties the picture to the video. Never write the "
            "tag <Picture 1> inside a field. Never write that the subject is shown, seen, visible, or "
            "pictured, and never write we see or there is. Write what the subject does."
        )
    elif task == "FL2V":
        lines.append("<Picture 1> is the exact first frame. <Picture 2> is the exact last frame.")
        lines.append("They are the two ends of one continuous shot. They are not two scenes.")
        lines.append(
            "Describe the subject once, as <Picture 1> shows it. Then describe the single continuous "
            "movement that carries it into the state <Picture 2> shows."
        )
        lines.append(
            "Every difference between the two pictures is an action or a camera movement. A wider "
            "end frame means the camera moved back. A different direction means the camera turned. "
            "A missing person walked out of the shot. A different place means the camera travelled "
            "there in one move. Different clothing or a different body means one thing changes into "
            "the other while the viewer watches."
        )
        lines.append("Never describe the two pictures as two separate states side by side. Describe the change between them.")
        lines.append(
            "The binding line above your answer already ties both pictures to the video. Never write the "
            "tag <Picture 1> or <Picture 2> inside a field. Never write that the subject is shown, seen, "
            "visible, or pictured, and never write we see or there is. Write what the subject does."
        )
    elif task in {"R2V", "R2VA"}:
        lines.append(
            "Each attached picture is an identity anchor. It fixes how a subject looks. It does not "
            "fix a moment in time and it does not fix the framing."
        )
        lines.append(
            "In subject_definitions, write one sentence for each picture. Start the sentence with that "
            "picture's tag, then the word is, then describe that subject in your own words: what kind of "
            "thing it is, and what it looks like. Write it in ordinary sentence case."
        )
        lines.append(
            "In the timeline, name the subject as a normal noun phrase and put its tag straight after that "
            "noun phrase, with a space between them. H3 binds a reference to a subject only through that "
            "inline tag. A subject with no tag loses its identity. Never fuse the tag onto a word."
        )
        lines.append(
            "In retention_analysis, name the exact traits that must not drift: the face, the hair colour and "
            "length, each piece of clothing, and the shape of any object. Name them. Do not write a general "
            "promise that things stay the same."
        )
        lines.append(
            "The pictures do not fix the setting. Build the setting, the light, and the action from "
            "the idea, and keep them consistent with the subjects you can see."
        )
    elif task in {"V2V", "V2VA"}:
        lines.append(
            "A reference video is attached. You are shown frames from it in time order. Those frames "
            "are one moving shot, not separate pictures."
        )
        lines.append("The reference video fixes the motion, the pacing, and the camera work. Follow its physical timing exactly.")
        lines.append("Write the tag <Video 1> inside the sentence that carries that motion.")
        if image_count:
            lines.append(
                "The attached picture(s) fix identity and look. Apply the face, the clothing, and the "
                "visual treatment of <Picture 1> to the motion taken from <Video 1>."
            )
        lines.append("Layer the style, mood, or setting the idea asks for on top of that physical continuity.")
    else:  # T2V with stray references
        lines.append("Treat any attached picture as loose visual guidance only. Build the scene from the idea.")

    if has_audio:
        lines.append(
            "A reference sound or voice track is attached as <Audio 1>. Describe how the subject's "
            "lips, body, or surroundings react to it, and bind it with the inline tag <Audio 1>."
        )
        lines.append(
            "When the subject speaks along with that track, say plainly that the lips move exactly "
            "in time with each word."
        )

    return "\n".join(lines)


# Ordered content slots. The list is a checklist of what to cover, not a
# sample of what to write. Longer videos take more of the list.
_COVERAGE = (
    "the medium and the overall light",
    "the subject, and exactly what it is wearing or made of",
    "the first thing the subject does",
    "what is directly behind the subject",
    "the colour and material of the nearest object",
    "how the light falls on the subject while it moves",
    "the ground or floor under the subject",
    "the second thing the subject does",
    "the surface texture of the subject, its skin, fur, fabric, or metal",
    "what is off to one side",
    "a small detail of the subject's face or hands",
    "something small drifting or moving in the air",
    "the third thing the subject does",
    "the quality of the air itself, clear, hazy, dusty, or misty",
    "what changes behind the subject as it moves",
    "what the camera does, ending with the exact amplitude and speed wording",
)


# Word and sentence budgets that hold up in practice, per clip length.
# Between the anchors the budget is interpolated.
_BUDGET_ANCHORS = ((2.0, 40, 4), (5.0, 75, 5), (8.0, 120, 10), (10.0, 160, 13), (15.0, 210, 16))


def budget_for(seconds: float) -> tuple[int, int]:
    """Return the word count and the sentence count for a duration."""
    value = snap_seconds(seconds)
    previous = _BUDGET_ANCHORS[0]
    for anchor in _BUDGET_ANCHORS:
        if value <= anchor[0]:
            if anchor[0] == previous[0]:
                return anchor[1], anchor[2]
            span = anchor[0] - previous[0]
            share = (value - previous[0]) / span
            words = previous[1] + (anchor[1] - previous[1]) * share
            sentences = previous[2] + (anchor[2] - previous[2]) * share
            return int(round(words)), int(round(sentences))
        previous = anchor
    return _BUDGET_ANCHORS[-1][1], _BUDGET_ANCHORS[-1][2]


def _length_block(seconds: float) -> str:
    """Build the word and coverage budget for a duration."""
    value = snap_seconds(seconds)
    words, sentences = budget_for(value)
    sentences = max(2, min(len(_COVERAGE), sentences))
    frames = int(round(value * H3_FPS))
    covered = _COVERAGE[: sentences - 1] + (_COVERAGE[-1],)
    checklist = "; ".join(covered)
    return f"""LENGTH.
The video is {value:g} seconds long at {H3_FPS} frames per second, which is about {frames} frames.
Write about {words} words in the timeline field, and never fewer.
Write at least {len(covered)} full sentences.
Give each of these its own sentence, in this order: {checklist}.
Keep writing until every item on that list has its own sentence. Do not stop early.
More words never means more actions. When you need more words, describe the same actions and the same place in more detail.
Count your words before you answer."""


def _alignment_note(line: str, schema: tuple[str, ...]) -> str:
    if not line:
        return ""
    return (
        "ALIGNMENT LINE.\n"
        "A binding line is added above your answer automatically. Do not write it yourself. "
        f"Do not write any line before {schema[0]}."
    )


def build_system_prompt(
    task: str,
    seconds: float,
    image_count: int,
    video_count: int,
    has_audio: bool,
    alignment: str = "",
) -> str:
    """Assemble the full system prompt for one MiniMax H3 task."""
    if task not in H3_TASKS:
        raise ValueError(f"Unknown MiniMax H3 task: {task}")

    schema = field_schema(task)
    blocks = [
        _ROLE,
        f"TASK.\n{H3_TASK_LABELS[task]}",
        _slot_block(task, image_count, video_count, has_audio),
        _output_contract(schema),
        _alignment_note(alignment, schema),
        _VIEWER_RULES,
        _STYLE_RULE,
        _SENTENCE_RULE,
        _MOTION_RULE,
        _CAMERA_RULE,
        _DIALOGUE_RULE,
        _TEXT_RULE,
        _SOUND_RULE,
        _length_block(seconds),
    ]
    return "\n\n".join(block for block in blocks if block)


def build_user_message(idea: str, task: str, image_count: int, video_count: int, has_audio: bool) -> str:
    """Wrap the user's rough idea with the binding tags that are available."""
    tags = [f"<Picture {index + 1}>" for index in range(image_count)]
    tags += [f"<Video {index + 1}>" for index in range(video_count)]
    if has_audio:
        tags.append("<Audio 1>")

    parts = [f"IDEA: {idea.strip()}"]
    if tags:
        parts.append(
            "AVAILABLE REFERENCE TAGS: "
            + ", ".join(tags)
            + ". Put each tag you use inside a sentence of the description."
        )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------

_FIELD_ALIASES = {
    "subject_definitions": "subject_definitions",
    "subjects": "subject_definitions",
    "summary": "summary",
    "retention_analysis": "retention_analysis",
    "retention": "retention_analysis",
    "integrated_multimodal_description": "integrated_multimodal_description",
    "detailed_description": "detailed_description",
    "description": "integrated_multimodal_description",
    "overall_soundscape": "overall_soundscape",
    "soundscape": "overall_soundscape",
    "audio": "overall_soundscape",
    "sound": "overall_soundscape",
    "sfx": "overall_soundscape",
    "non_diegetic_music": "non_diegetic_music",
    "music": "non_diegetic_music",
    "score": "non_diegetic_music",
    "soundtrack": "non_diegetic_music",
}




# Hedging adverbs weaken a video prompt and the model reaches for them anyway.
# Deleting a lone adverb never breaks the sentence around it.
_HEDGE_WORDS = ("slightly", "subtly", "gently", "gradually", "somewhat", "a little", "a bit", "slight")


def scrub_hedges(text: str) -> str:
    """Delete hedging adverbs that the rules already forbid."""
    import re

    for word in _HEDGE_WORDS:
        text = re.sub(rf"\b{re.escape(word)}\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"[ \t]{2,}", " ", text)


def _strip_fences(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def parse_fields(text: str, schema: tuple[str, ...] = _SIMPLE_FIELDS) -> dict[str, str]:
    """Split a model answer into the H3 fields of one schema."""
    body_field = _body_field(schema)
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in _strip_fences(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        head, separator, tail = stripped.partition(":")
        key = head.strip().lower().replace(" ", "_").lstrip("*#- ").strip()
        if separator and key in _FIELD_ALIASES:
            current = _FIELD_ALIASES[key]
            fields.setdefault(current, [])
            if tail.strip():
                fields[current].append(tail.strip())
            continue
        if current is None:
            current = body_field
            fields.setdefault(current, [])
        fields[current].append(stripped)
    return {key: " ".join(value).strip() for key, value in fields.items() if " ".join(value).strip()}


def compile_prompt(raw: str, alignment: str = "", schema: tuple[str, ...] = _SIMPLE_FIELDS) -> str:
    """Repair a model answer into the exact layout H3 expects."""
    body_field = _body_field(schema)
    fields = parse_fields(raw or "", schema)

    # A model that answered with the other schema's timeline name still gets
    # its timeline used, rather than dropped on the floor.
    if not fields.get(body_field):
        for spare in ("detailed_description", "integrated_multimodal_description"):
            if fields.get(spare):
                fields[body_field] = fields.pop(spare)
                break

    for key, value in list(fields.items()):
        fields[key] = scrub_hedges(value)

    body = fields.get(body_field, "").strip()
    if body and "[shot 1]" not in body.lower():
        body = f"[Shot 1] {body}"
    if body:
        fields[body_field] = body

    parts = []
    if alignment:
        parts.append(alignment)
    for key in schema:
        value = fields.get(key, "").strip()
        if value:
            parts.append(f"{key}: {value}")

    compiled = "\n\n".join(parts).strip()
    if len(compiled) > H3_MAX_CHARS:
        compiled = compiled[:H3_MAX_CHARS].rstrip()
    return compiled
