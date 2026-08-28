"""MiniMax H3 prompt modes for the VLM Prompt Enhancer node.

This module holds everything that is specific to the MiniMax H3 video model:

* the generation modes the model accepts, under MiniMax's own names,
* automatic mode detection from the connected reference inputs,
* the system prompt that is built for each mode,
* the alignment line that binds a reference picture to a point in time,
* a post-processor that repairs the field layout H3 expects.

The system prompts contain rules only. They contain no example scene, no
example subject, and no example finished prompt. A language model copies what
it is shown, so showing it a finished prompt biases every later generation
toward that content.

H3 always produces audio with the video. That is why every mode name ends in
A. Where MiniMax's own names differ from the shorter names in common use, the
MiniMax name wins, and the short name is kept as a silent alias so that saved
workflows keep loading.
"""

from __future__ import annotations

import re

# The cloud endpoint accepts at most this many characters per text item.
H3_MAX_CHARS = 7000

MODE_DEFAULT = "Default"
MODE_H3 = "MiniMax H3"
MODES = (MODE_DEFAULT, MODE_H3)

H3_AUTO = "Auto"
H3_TASKS = ("T2VA", "I2VA", "L2VA", "FL2VA", "Ref2VA")
H3_TASK_OPTIONS = (H3_AUTO,) + H3_TASKS

# Older widget values that saved workflows may still carry.
_TASK_ALIASES = {
    "T2V": "T2VA",
    "I2V": "I2VA",
    "L2V": "L2VA",
    "FL2V": "FL2VA",
    "R2V": "Ref2VA",
    "R2VA": "Ref2VA",
    "V2V": "Ref2VA",
    "V2VA": "Ref2VA",
}

H3_TASK_LABELS = {
    "T2VA": "Text to video. No reference media.",
    "I2VA": "Image to video. One picture is the exact first frame.",
    "L2VA": "Last frame to video. One picture is the exact last frame.",
    "FL2VA": "First and last frame to video. Two pictures bound one shot.",
    "Ref2VA": "Reference to video. Pictures, videos, and audio carry identity, motion, and sound.",
}

# Reference-task kinds. MiniMax writes these in square brackets in front of
# the summary line, and allows more than one, joined by " + ".
REF_AUTO = "Auto"
REF_KINDS = (
    "reference generation",
    "keyframe completion",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)
REF_KIND_OPTIONS = (REF_AUTO,) + REF_KINDS

# H3 reads a small set of named fields. Reference tasks carry the longer set,
# because each subject must be declared before the timeline starts.
_SIMPLE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
_REFERENCE_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

# Generation grid and duration range. The cloud endpoint takes a whole number
# of seconds from 4 to 15. The ComfyUI nodes snap the frame count to a 17k+5
# grid at 24 frames per second, whose lowest trained value is 124 frames.
H3_FPS = 24
H3_MIN_SECONDS = 4
H3_MAX_SECONDS = 15

# Reference capacity of one request.
H3_MAX_REF_IMAGES = 9
H3_MAX_REF_VIDEOS = 3
H3_MAX_REF_AUDIO = 3

# Modes that must stay inside one continuous shot.
_SINGLE_SHOT_TASKS = frozenset({"I2VA", "L2VA", "FL2VA"})


# Quotation marks people actually type, straight and curly, plus the Japanese
# corner brackets MiniMax's own guide uses.
_QUOTE_PAIRS = (('"', '"'), ("\u201c", "\u201d"), ("\u2018", "\u2019"), ("\u300c", "\u300d"), ("'", "'"))
_MUSIC_WORDS = (
    "music", "score", "soundtrack", "song", "singing", "sings", "sing", "melody",
    "piano", "guitar", "violin", "strings", "synth", "drums", "beat", "orchestra",
    "jazz", "lo-fi", "lofi", "ambient track", "background track", "theme tune",
)


def extract_spoken_lines(idea: str) -> list[str]:
    """Return the quoted lines in an idea, in the order they were written.

    Quoted text is the only reliable signal that the user wants speech. A
    request such as "she talks about her day" describes an action; it hands
    over no words, so nothing can be said.
    """
    text = idea or ""
    found: list[str] = []
    for opening, closing in _QUOTE_PAIRS:
        start = 0
        while True:
            left = text.find(opening, start)
            if left < 0:
                break
            right = text.find(closing, left + len(opening))
            if right < 0:
                break
            line = text[left + len(opening):right].strip()
            start = right + len(closing)
            if len(line) >= 2 and line not in found:
                found.append(line)
    return found


def wants_music(idea: str) -> bool:
    """Say whether the idea actually asked for a musical score."""
    lowered = (idea or "").lower()
    return any(word in lowered for word in _MUSIC_WORDS)


def speech_sentence(line: str, language: str = "English") -> str:
    """Build one correctly shaped spoken passage for H3."""
    words = line.strip()
    if words and words[-1] not in ".?!":
        words += "."
    if words:
        words = words[0].upper() + words[1:]
    return (
        "The subject's jaw and lips move clearly through every word, and the subject (S1) says: "
        f"<d>[{language}] {words}</d> The subject closes their lips."
    )


def field_schema(task: str) -> tuple[str, ...]:
    """Return the field names, in order, that H3 expects for a task."""
    return _REFERENCE_FIELDS if task == "Ref2VA" else _SIMPLE_FIELDS


def _body_field(schema: tuple[str, ...]) -> str:
    """Return the field that holds the timeline."""
    return "detailed_description" if "detailed_description" in schema else "integrated_multimodal_description"


def detect_task(image_count: int, has_video: bool, has_audio: bool) -> str:
    """Choose a task code from the reference media that is connected."""
    if has_video or has_audio or image_count >= 3:
        return "Ref2VA"
    if image_count == 2:
        return "FL2VA"
    if image_count == 1:
        return "I2VA"
    return "T2VA"


def resolve_task(selection: str, image_count: int, has_video: bool, has_audio: bool) -> str:
    """Return the task code to use for this run."""
    choice = (selection or H3_AUTO).strip()
    if choice and choice != H3_AUTO:
        choice = _TASK_ALIASES.get(choice, choice)
        if choice not in H3_TASKS:
            raise ValueError(f"Unknown MiniMax H3 task: {selection}")
        return choice
    return detect_task(image_count, has_video, has_audio)


def expected_image_count(task: str) -> tuple[int, int]:
    """Return the smallest and largest picture count a task accepts."""
    if task == "T2VA":
        return (0, 0)
    if task in {"I2VA", "L2VA"}:
        return (1, 1)
    if task == "FL2VA":
        return (2, 2)
    return (0, H3_MAX_REF_IMAGES)


def check_image_count(task: str, image_count: int) -> None:
    """Reject a picture count that the task cannot use."""
    low, high = expected_image_count(task)
    if low <= image_count <= high:
        return
    wanted = f"exactly {low}" if low == high else f"{low} to {high}"
    raise ValueError(
        f"MiniMax H3 task {task} takes {wanted} picture(s), but {image_count} were connected. "
        "Use first_frame, last_frame, and reference_images to set the picture order."
    )


def snap_seconds(seconds) -> int:
    """Clamp a requested duration to a whole second H3 accepts."""
    value = int(round(float(seconds)))
    return max(H3_MIN_SECONDS, min(H3_MAX_SECONDS, value))


def resolve_ref_kinds(
    selection: str, image_count: int, has_video: bool, has_audio: bool, has_keyframe: bool
) -> str:
    """Build the bracketed reference-task label that H3 reads on the summary line."""
    choice = (selection or REF_AUTO).strip()
    if choice and choice != REF_AUTO:
        return choice

    kinds = []
    if image_count or has_video:
        # A reference video that only supplies camera work or rhythm is still
        # reference generation. Video editing is a stronger claim, so it is
        # never assumed.
        kinds.append("reference generation")
    if has_keyframe:
        kinds.append("keyframe completion")
    if has_audio:
        kinds.append("audio reference")
    return " + ".join(kinds)


def alignment_line(task: str, image_count: int, seconds, anchor_first_reference: bool) -> str:
    """Build the reference-to-timeline binding line that H3 reads first.

    The node writes this line itself. A small model cannot be trusted to copy
    a fixed string and a formatted number without drift. MiniMax words the
    line differently for each mode, and those differences are kept.
    """
    if image_count <= 0:
        return ""
    end = "%.2f" % float(snap_seconds(seconds))
    if task == "I2VA":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if task == "L2VA":
        return (
            "How the reference pictures align with the target video - <Picture 1> (from [Shot 1]) "
            f"aligns with the {end}-second mark of the target video."
        )
    if task == "FL2VA" and image_count >= 2:
        return (
            "How the reference pictures align with the target video - "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot 1) aligns with the {end}-second mark of the target video."
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
    "subject_definitions": "<declare every subject, one sentence each>",
    "summary": "<the bracketed task label, then one sentence that says what the video shows>",
    "retention_analysis": "<one line for each reference, naming how strongly it is kept>",
    "detailed_description": "[Shot 1] <the timeline>",
    "integrated_multimodal_description": "[Shot 1] <the timeline>",
    "overall_soundscape": "<the real sounds of that place>",
    "non_diegetic_music": "<the score the audience hears, or N/A>",
}


def _output_contract(schema: tuple[str, ...]) -> str:
    shape = "\n\n".join(f"{name}: {_FIELD_HELP[name]}" for name in schema)
    return f"""OUTPUT SHAPE.
Your whole answer is these {len(schema)} fields, in this order, and nothing else:

{shape}

Put one empty line between the fields.
Never put a line break inside a field.
Never write a heading, a title, a bullet, a number list, a code fence, or quotation marks around a field.
Never write a greeting, a preface, a note, or a closing remark. Write nothing outside the fields.
Never repeat these instructions back.
Never write your reasoning. Write the fields only.
Your answer ends at the end of the {schema[-1]} line.
Write in English only. Never write Chinese characters.

FIELD NAMES.
Every field begins with its exact name, then a colon, then a space. Type the name even when the field is short.
Your answer contains all {len(schema)} field names, each one time, in the order above.
Never merge two fields into one. Never let the content of one field run on at the end of another field.
Never leave a field out. A field with nothing to say still gets its name, followed by N/A."""


_VIEWER_RULES = """WHAT THE VIEWER SEES.
Write only what a viewer can see or hear. Never write a smell, a taste, an emotion, or a thought.
The viewer watches a moving video. The words picture, image, photo, photograph, frame, reference, left, right, side by side, split screen and collage never appear inside any field.
Some words in the idea name the camera, not a thing in the video. Words such as drone, aerial, bird's eye, crane, helicopter, close-up and wide shot describe how the shot is taken. The viewer never sees the camera or the operator. Turn such a word into a camera movement, and never put the device in the scene.
Never open the description by repeating the idea sentence.
Describe what is there. Never describe what is absent, missing, or must not appear. Naming an unwanted thing can put it on screen."""

_STYLE_RULE = """STYLE SENTENCE.
Begin the timeline with one short style sentence. Name the medium and the overall light.
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
Never write camera words as a tag list at the end. Never put a camera term in square brackets.
Use at most two camera movements in the whole timeline.
Use only these movements, and keep their wording: Zoom In, Zoom Out, Push In, Pull Out, Pan Left, Pan Right, Truck Left, Truck Right, Tilt Up, Tilt Down, Pedestal Up, Pedestal Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly, Shake Strongly, POV, Roll Clockwise, Roll Counterclockwise.
Zoom changes the lens. Push and Pull move the camera body. Pick the one you mean.
Every camera movement ends with an amplitude phrase and then a speed phrase, in this exact wording: "with small amplitude at slow speed", "with small amplitude at fast speed", "with large amplitude at slow speed", or "with large amplitude at fast speed".
A calm scene takes small amplitude and slow speed. Only a genuinely energetic scene takes large amplitude or fast speed.
These words are banned everywhere: subtly, gently, gradually, somewhat, a little."""

_SINGLE_SHOT_RULE = """SHOTS.
The whole timeline is one continuous shot. It holds [Shot 1] and nothing else.
Never write a cut, a transition, a montage, a second scene, or a second shot."""

_MULTI_SHOT_RULE = """SHOTS.
One shot is normal. Write a second shot only when the idea clearly asks for a cut.
When you do write more than one shot, start each one with its label and its cut time, in this shape: [Shot N] At MM:SS.mmm, and let the cut times rise through the timeline.
Never write a montage or a transition effect."""

_MOTION_RULE = """MOTION.
Something must move in every second of the video. A still description makes a still video.
Write motion as one continuous, physically possible chain. Each action causes the next one.
Match the amount of action to the clip length. Fewer actions described fully beat many actions rushed.
Never write a speed change such as slow motion or time lapse unless the idea asks for it."""

_TEXT_RULE = """ON-SCREEN TEXT.
Any words that must appear on screen, on a sign or a title, go inside straight double quotation marks. Write nothing else in quotation marks.
Keep on-screen words in the language the user wrote them in. Never translate them."""

_DIALOGUE_RULE = """SPEECH.
Quoted words in the idea are lines that MUST be spoken on screen. They are not a description. You must place every one of them in the timeline, inside the wrapper below, word for word.
Never replace a quoted line with a description such as she speaks, she talks, or she tells the viewer. A description makes the video silent, which is a failure.
When the idea hands you no quoted words, nobody speaks: write no speech at all.
When somebody does speak, the spoken words appear one time only, and they appear inside this wrapper: <d>[English] The words.</d>
Copy the square brackets and the tag exactly. Change English to the language of the words when the user wrote them in another language, and never translate them.
The words start with a capital letter and end with a full stop, a question mark, or an exclamation mark.
Build the spoken passage in this order, writing every part in your own words and in ordinary sentence case:
1. Say that the speaker's jaw and lips move clearly through every word.
2. Name the speaker, then the quality of their voice, then the speaker tag in round brackets.
3. Write the word says, a colon, and then the wrapper holding the exact words.
4. Say that the speaker closes their lips.
5. Add one small action, and only one.
Use (S1) for the first speaker and (S2) for a second speaker, in the order they speak.
Match the pronoun to the person you can see. Never write his or her, and never leave a description of the voice unwritten.
Speech takes real time. Keep the spoken words short enough to be said inside the clip. Cut a line rather than let the delivery rush.
Never summarise, hint at, or repeat the spoken words anywhere else."""

_SOUND_RULE = """SOUND FIELDS.
overall_soundscape lists three to five real sounds of the place, and nothing else.
No person makes a vocal sound in overall_soundscape. These words are banned from that field: voice, voices, talking, talk, chatter, murmur, mutter, conversation, speech, speaking, whisper, whispering, shouting, singing, lyrics, crowd noise.
A crowded place is still silent of people in this field. Name objects instead of people, such as cups, doors, machines, footsteps, or traffic.
Moving air, cloth, and machines never whisper or murmur. They hiss, rush, hum, rustle, creak, or sigh.
Never ask for silence or for clean audio. A room always has room tone.
non_diegetic_music is "N/A" by default. Write a score ONLY when the idea asks for music, a song, or an instrument by name. Never add music because the scene feels like it wants some. Unwanted music ruins the take.
When the idea does ask for music, name instruments, tempo, and volume."""

_NO_TAG_RULE = (
    "The binding line above your answer already ties the picture to the video. Never write a "
    "<Picture> tag inside a field. Never write that the subject is shown, seen, visible, or "
    "pictured, and never write we see or there is. Write what the subject does."
)


def _slot_block(
    task: str,
    image_count: int,
    video_count: int,
    audio_count: int,
    ref_kinds: str,
    video_frame_count: int = 0,
) -> str:
    """Explain what each attached reference means for this task.

    ``video_frame_count`` counts pictures at the end of the attached set that
    are consecutive frames of a reference video rather than separate subjects.
    Passing a video as ordered frames works on every vision model; the
    dedicated video path of a processor does not.
    """
    lines = ["REFERENCE MEDIA."]

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

    if video_frame_count > 0 and image_count >= video_frame_count:
        first = image_count - video_frame_count + 1
        names = f"<Picture {first}>" if video_frame_count == 1 else f"<Picture {first}> to <Picture {image_count}>"
        lines.append(
            f"The last {video_frame_count} of those pictures, {names}, are not separate subjects. They are "
            "consecutive frames of one reference video, in time order. Read them as one moving shot. "
            "Refer to that shot as <Video 1>, and never refer to those frames by their picture numbers."
        )

    if task == "I2VA":
        lines.append("<Picture 1> is the exact first frame of the video. The video starts inside it.")
        lines.append(
            "Anchor the opening in one or two sentences: the same person or object, the same clothing, "
            "the same place, and the same light. Then move forward in time."
        )
        lines.append("Never invent a different scene. Never change the clothing. Never add a person who is not there.")
        lines.append(
            "Appearance is already fixed, so spend the rest of your words on movement, on physics, and "
            "on what changes over time."
        )
        lines.append(_NO_TAG_RULE)
    elif task == "L2VA":
        lines.append("<Picture 1> is the exact last frame of the video. The video ends inside it.")
        lines.append(
            "Work backwards. Invent a believable earlier state that the same subject, the same clothing, "
            "and the same place could have been in a few seconds before."
        )
        lines.append(
            "Then write the single continuous movement that carries that earlier state into the final "
            "state. The final state is fixed, so describe arriving at it, not leaving it."
        )
        lines.append(_NO_TAG_RULE)
    elif task == "FL2VA":
        lines.append("<Picture 1> is the exact first frame. <Picture 2> is the exact last frame.")
        lines.append("They are the two ends of one continuous shot. They are not two scenes.")
        lines.append(
            "Describe the subject once, as the first frame shows it. Then describe the single continuous "
            "movement that carries it into the state the last frame shows."
        )
        lines.append(
            "Every difference between the two frames is an action or a camera movement. A wider end "
            "frame means the camera moved back. A different direction means the camera turned. A "
            "missing person walked out of the shot. A different place means the camera travelled there "
            "in one move. Different clothing or a different body means one thing changes into the other "
            "while the viewer watches."
        )
        lines.append("Never describe the two states side by side. Describe the change between them.")
        lines.append(_NO_TAG_RULE)
    elif task == "Ref2VA":
        if image_count:
            lines.append(
                "Each attached picture is an identity anchor. It fixes how a subject looks. It does not "
                "fix a moment in time and it does not fix the framing."
            )
        if video_count:
            names = ", ".join(f"<Video {index + 1}>" for index in range(video_count))
            lines.append(
                f"{video_count} reference video(s) are attached as {names}. You are shown their frames in "
                "time order. Those frames are one moving shot, not separate pictures. A reference video "
                "fixes motion, pacing, and camera work."
            )
            lines.append(
                "Describe the movement, the pacing, and the camera work you can read from those frames, "
                "and bind them with the inline tag <Video 1>. The subject that performs that movement is "
                "the one the pictures fix, not the person in the video frames."
            )
        if audio_count:
            names = ", ".join(f"<Audio {index + 1}>" for index in range(audio_count))
            lines.append(
                f"{audio_count} reference sound or voice track(s) are attached as {names}. Describe how "
                "the subject's lips, body, or surroundings react to the sound, and say plainly when the "
                "lips move exactly in time with each word."
            )
        lines.append(
            "In subject_definitions, declare each subject once. Start the sentence with a subject label, "
            "then the word is, then describe that subject in your own words, then say which reference it "
            "comes from by writing that reference's tag in the same sentence. Label the first subject "
            "<Subject 1>, the second <Subject 2>, and so on. Write it in ordinary sentence case."
        )
        lines.append(
            "In the timeline, name the subject as a normal noun phrase and put its subject label straight "
            "after that noun phrase, with a space between them. H3 binds a reference to a subject only "
            "through that inline label. A subject with no label loses its identity. Never fuse a label "
            "onto a word."
        )
        lines.append(
            "In retention_analysis, write one line for each reference. Start with the subject label, then "
            "in round brackets say which shots it appears in, then a colon, then one of these exact words "
            "for a picture or video reference: fully_preserved, partially_preserved, attribute_transfer, "
            "weak_reference. For an audio reference use one of these exact words instead: fully_copy, "
            "partially_copy, reference, weak_reference. After that word write a hyphen, then name the "
            "exact traits that must not drift, such as the face, the hair colour and length, each piece "
            "of clothing, and the shape of an object. Never write a speaker tag such as (S1) in this field."
        )
        if ref_kinds:
            lines.append(
                f"Begin the summary field with this exact label, square brackets included: [{ref_kinds}] "
                "Then write the one-sentence summary after it."
            )
        lines.append(
            "The references do not fix the setting. Build the setting, the light, and the action from the "
            "idea, and keep them consistent with the subjects you can see."
        )
    else:  # T2VA
        lines.append("No reference media is attached. You invent the whole scene from the idea.")
        lines.append(
            "Because nothing is fixed for you, describe the subject, the clothing, the surface materials, "
            "the setting, and the light in full. Leave nothing to chance."
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

# Word and sentence budgets per clip length, measured against what a small
# local model reliably produces. Between the anchors the budget is
# interpolated. MiniMax publishes a longer 350 to 500 word range for reference
# tasks, which _length_block applies on top of these numbers.
_BUDGET_ANCHORS = ((4, 65, 5), (5, 75, 5), (8, 120, 10), (10, 160, 13), (15, 210, 16))
_REFERENCE_WORD_FLOOR = 350
_REFERENCE_WORD_CEILING = 500


def budget_for(seconds) -> tuple[int, int]:
    """Return the word count and the sentence count for a duration."""
    value = snap_seconds(seconds)
    previous = _BUDGET_ANCHORS[0]
    for anchor in _BUDGET_ANCHORS:
        if value <= anchor[0]:
            if anchor[0] == previous[0]:
                return anchor[1], anchor[2]
            share = (value - previous[0]) / (anchor[0] - previous[0])
            words = previous[1] + (anchor[1] - previous[1]) * share
            sentences = previous[2] + (anchor[2] - previous[2]) * share
            return int(round(words)), int(round(sentences))
        previous = anchor
    return _BUDGET_ANCHORS[-1][1], _BUDGET_ANCHORS[-1][2]


def _length_block(seconds, task: str, spoken_lines: int = 0) -> str:
    """Build the word and coverage budget for a duration."""
    value = snap_seconds(seconds)
    words, sentences = budget_for(value)
    ceiling = ""
    if task == "Ref2VA":
        # MiniMax's own guidance for reference tasks. The clip length still
        # decides how much action fits; only the word floor changes.
        words = max(words, _REFERENCE_WORD_FLOOR)
        ceiling = f"Do not go past {_REFERENCE_WORD_CEILING} words.\n"
    sentences = max(2, min(len(_COVERAGE), sentences))
    frames = value * H3_FPS
    covered = _COVERAGE[: sentences - 1] + (_COVERAGE[-1],)
    checklist = "; ".join(covered)
    speech = ""
    if spoken_lines:
        speech = (
            f"On top of that list, write {spoken_lines} spoken passage(s), one for each quoted line "
            "the idea gave you, in the shape the SPEECH rule sets out. The spoken passages are "
            "required. They do not replace any item on the list.\n"
        )
    return f"""LENGTH.
The video is {value} seconds long at {H3_FPS} frames per second, which is {frames} frames.
Write about {words} words in the timeline field, and never fewer.
{ceiling}Write at least {len(covered)} full sentences.
Give each of these its own sentence, in this order: {checklist}.
{speech}Keep writing until every item on that list has its own sentence. Do not stop early.
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
    seconds,
    image_count: int,
    video_count: int = 0,
    audio_count: int = 0,
    alignment: str = "",
    ref_kinds: str = "",
    video_frame_count: int = 0,
    spoken_lines: int = 0,
) -> str:
    """Assemble the full system prompt for one MiniMax H3 task."""
    if task not in H3_TASKS:
        raise ValueError(f"Unknown MiniMax H3 task: {task}")

    schema = field_schema(task)
    blocks = [
        _ROLE,
        f"TASK.\n{H3_TASK_LABELS[task]}",
        _slot_block(task, image_count, video_count, audio_count, ref_kinds, video_frame_count),
        _output_contract(schema),
        _alignment_note(alignment, schema),
        _VIEWER_RULES,
        _STYLE_RULE,
        _SENTENCE_RULE,
        _SINGLE_SHOT_RULE if task in _SINGLE_SHOT_TASKS else _MULTI_SHOT_RULE,
        _MOTION_RULE,
        _CAMERA_RULE,
        _DIALOGUE_RULE,
        _TEXT_RULE,
        _SOUND_RULE,
        _length_block(seconds, task, spoken_lines),
    ]
    return "\n\n".join(block for block in blocks if block)


def build_user_message(
    idea: str,
    task: str,
    image_count: int,
    video_count: int = 0,
    audio_count: int = 0,
    video_frame_count: int = 0,
    spoken_lines: tuple[str, ...] = (),
) -> str:
    """Wrap the user's rough idea with the binding tags that are available."""
    idea = (idea or "").strip()
    if idea:
        parts = [f"IDEA: {idea}"]
    else:
        # No idea was typed. The reference media is the whole brief, so say so
        # plainly rather than leaving the model an empty instruction to fill.
        parts = [
            "IDEA: none given. Build the video from the reference media alone. "
            "Keep the subject, the setting, and the look that the reference media shows, "
            "and invent only the movement that suits it."
        ]
    if spoken_lines:
        quoted = "\n".join(f"  {index + 1}. {line}" for index, line in enumerate(spoken_lines))
        parts.append(
            "REQUIRED SPOKEN LINES. The idea quoted these words. Each one MUST be spoken on screen, "
            "word for word, inside a <d> wrapper, in the shape the SPEECH rule sets out:\n"
            f"{quoted}\n"
            "Do not paraphrase them. Do not summarise them. Do not replace them with a description "
            "of somebody speaking."
        )
    if task == "Ref2VA":
        tags = [f"<Picture {index + 1}>" for index in range(image_count - video_frame_count)]
        tags += [f"<Video {index + 1}>" for index in range(video_count)]
        tags += [f"<Audio {index + 1}>" for index in range(audio_count)]
        if tags:
            parts.append(
                "AVAILABLE REFERENCE TAGS: "
                + ", ".join(tags)
                + ". Bind each one in subject_definitions and in retention_analysis."
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

# Hedging adverbs weaken a video prompt and a model reaches for them anyway.
# Deleting a lone adverb never breaks the sentence around it. "slightly" is
# handled apart, because "Shake Slightly" is an official camera term.
_HEDGE_WORDS = ("subtly", "gently", "gradually", "somewhat", "a little", "a bit", "slight")
_HEDGE_SLIGHT = re.compile(r"(?<!shake )\bslightly\s+", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove a leaked chain-of-thought block from a thinking model."""
    cleaned = _THINK_BLOCK.sub("", text or "")
    # An unterminated opening tag means the answer proper starts after it.
    tail = re.split(r"</think>", cleaned, flags=re.IGNORECASE)
    return tail[-1].strip() if len(tail) > 1 else cleaned.strip()


def scrub_hedges(text: str) -> str:
    """Delete hedging adverbs that the rules already forbid."""
    for word in _HEDGE_WORDS:
        text = re.sub(rf"\b{re.escape(word)}\s+", "", text, flags=re.IGNORECASE)
    text = _HEDGE_SLIGHT.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text)


# MiniMax writes a subject label in angle brackets, and separates a retention
# marker from its detail with a spaced hyphen. Small models drop both.
_BARE_SUBJECT = re.compile(r"(?<![<\w])Subject (\d+)")
_RETENTION_MARKERS = (
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
    "fully_copy",
    "partially_copy",
)


def repair_reference_labels(text: str) -> str:
    """Restore the angle brackets and marker spacing H3 expects."""
    text = _BARE_SUBJECT.sub(r"<Subject \1>", text)
    for marker in _RETENTION_MARKERS:
        text = re.sub(rf"\b{marker}\s*-\s*", f"{marker} - ", text)
    return text


# Wording a model copies straight out of an instruction instead of replacing.
# H3 would read it as literal on-screen direction, so it is removed.
_PLACEHOLDER_FIXES = (
    (re.compile(r"\bhis or her\b", re.IGNORECASE), "the speaker's"),
    (re.compile(r"\bwith a VOICE QUALITY voice\b"), "with a clear voice"),
    (re.compile(r"\bVOICE QUALITY\b"), "clear"),
    (re.compile(r"\bTHE WORDS\b"), ""),
    (re.compile(r"\band ONE ACTION\b"), ""),
    (re.compile(r"\bONE ACTION\b"), ""),
    (re.compile(r"\bA PLAIN DESCRIPTION OF THAT SUBJECT\b"), ""),
    (re.compile(r"\bSPEAKER\b"), "the speaker"),
)


def scrub_placeholders(text: str) -> str:
    """Remove instruction placeholders a model copied verbatim."""
    for pattern, replacement in _PLACEHOLDER_FIXES:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _strip_fences(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def parse_fields(text: str, schema: tuple[str, ...] = _SIMPLE_FIELDS) -> dict[str, str]:
    """Split a model answer into the H3 fields of one schema."""
    body_field = _body_field(schema)
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in _strip_fences(strip_reasoning(text)).splitlines():
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


_D_BLOCK = re.compile(r"<d>(.*?)</d>", re.DOTALL | re.IGNORECASE)
_D_ANY = re.compile(r"</?d>", re.IGNORECASE)
_LANG_PREFIX = re.compile(r"^\s*\[[A-Za-z ]+\]")
# A spoken line is a line. Anything longer than this is the model having run
# the rest of the timeline into an unclosed wrapper.
_MAX_SPOKEN_CHARS = 300


def strip_broken_wrappers(body: str) -> str:
    """Remove every <d> marker when the wrappers cannot be trusted.

    An unclosed <d>, or one holding a whole paragraph, would otherwise pair
    with the next closing tag and swallow the timeline. Clearing the markers
    lets the required lines be put back cleanly.
    """
    opens = len(re.findall(r"<d>", body, re.IGNORECASE))
    closes = len(re.findall(r"</d>", body, re.IGNORECASE))
    oversized = any(len(m) > _MAX_SPOKEN_CHARS for m in _D_BLOCK.findall(body))
    if opens != closes or oversized:
        return _D_ANY.sub("", body)
    return body


def _signature(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def normalise_speech_blocks(body: str, spoken_lines: tuple[str, ...] = (), language: str = "English") -> str:
    """Make every <d> wrapper hold the spoken words and nothing else.

    H3 reads the wrapper as the literal line to say. A model often stuffs the
    speaker, the voice, and the speaker tag inside it, or drops the language
    tag. Either way the take comes out wrong, so the wrapper is rebuilt.
    """
    known = {_signature(line): line for line in spoken_lines}

    def rebuild(match):
        content = match.group(1).strip()
        signature = _signature(content)
        for sig, line in known.items():
            if sig and sig in signature:
                words = line.strip().strip('"\u201c\u201d\u2018\u2019')
                if words and words[-1] not in ".?!":
                    words += "."
                if words:
                    words = words[0].upper() + words[1:]
                return f"<d>[{language}] {words}</d>"
        # An unrecognised line still needs its language tag and clean edges.
        cleaned = content.strip().strip('"\u201c\u201d\u2018\u2019').strip()
        if not _LANG_PREFIX.match(cleaned):
            cleaned = f"[{language}] {cleaned}"
        return f"<d>{cleaned}</d>"

    return _D_BLOCK.sub(rebuild, body)


def _has_spoken(body: str, line: str) -> bool:
    """Say whether a quoted line already sits inside a <d> wrapper."""
    needle = re.sub(r"[^a-z0-9]+", "", line.lower())
    if not needle:
        return True
    for block in re.findall(r"<d>(.*?)</d>", body, flags=re.DOTALL | re.IGNORECASE):
        if needle in re.sub(r"[^a-z0-9]+", "", block.lower()):
            return True
    return False


def compile_prompt(
    raw: str,
    alignment: str = "",
    schema: tuple[str, ...] = _SIMPLE_FIELDS,
    spoken_lines: tuple[str, ...] = (),
    allow_music: bool = True,
) -> str:
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

    reference_schema = "subject_definitions" in schema
    for key, value in list(fields.items()):
        value = scrub_placeholders(scrub_hedges(value))
        if reference_schema:
            value = repair_reference_labels(value)
        fields[key] = value

    body = fields.get(body_field, "").strip()
    if body and "[shot 1]" not in body.lower():
        body = f"[Shot 1] {body}"

    # Rebuild every wrapper before checking, so a line buried inside a
    # malformed wrapper counts as spoken and is not added a second time.
    if body:
        body = strip_broken_wrappers(body)
        body = normalise_speech_blocks(body, spoken_lines)

    # A small model often narrates that somebody speaks instead of quoting the
    # words. H3 then generates a silent take. Put any dropped line back.
    for line in spoken_lines:
        if body and not _has_spoken(body, line):
            body = f"{body.rstrip()} {speech_sentence(line)}"

    if body:
        fields[body_field] = body

    # Music arrives only on request. An invented score cannot be removed from
    # a finished generation.
    if not allow_music:
        fields["non_diegetic_music"] = "N/A"

    def assemble(values: dict[str, str]) -> str:
        parts = [alignment] if alignment else []
        parts += [f"{key}: {values[key].strip()}" for key in schema if values.get(key, "").strip()]
        return "\n\n".join(parts).strip()

    compiled = assemble(fields)
    if len(compiled) <= H3_MAX_CHARS:
        return compiled

    # Trim the timeline rather than the tail, so the sound fields survive.
    overflow = len(compiled) - H3_MAX_CHARS
    trimmed = dict(fields)
    trimmed[body_field] = body[: max(0, len(body) - overflow)].rstrip()
    return assemble(trimmed)[:H3_MAX_CHARS].rstrip()
