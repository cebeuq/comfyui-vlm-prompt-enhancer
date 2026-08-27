from __future__ import annotations


def _sample_indices(length: int, maximum: int) -> list[int]:
    if length <= 0:
        return []
    count = min(length, maximum)
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def tensor_to_pil_images(images, maximum: int = 8):
    """Convert a ComfyUI BHWC float tensor to a small list of PIL images."""
    from PIL import Image

    output = []
    for index in _sample_indices(int(images.shape[0]), maximum):
        array = images[index].detach().cpu().float().clamp(0, 1).numpy()
        array = (array * 255.0).round().astype("uint8")
        output.append(Image.fromarray(array[..., :3], mode="RGB"))
    return output


def video_to_array(video, maximum: int = 16):
    """Decode and uniformly sample a native ComfyUI VIDEO object."""
    import numpy as np

    if not hasattr(video, "get_components"):
        raise TypeError("The video input must be a native ComfyUI VIDEO object")
    frames = video.get_components().images
    sampled = tensor_to_pil_images(frames, maximum)
    if not sampled:
        raise ValueError("The video contains no frames")
    return np.stack([np.asarray(frame) for frame in sampled])


def build_messages(system_prompt: str, prompt: str, images=None, video=None) -> list[dict]:
    content = []
    for image in images or []:
        content.append({"type": "image", "image": image})
    if video is not None:
        content.append({"type": "video", "video": video})
    content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": content})
    return messages


def collect_images(
    first_frame=None,
    last_frame=None,
    reference_images=None,
    maximum: int = 9,
    extra_images=(),
):
    """Order the reference pictures the way MiniMax H3 numbers them.

    Slot order is fixed: the first-frame input becomes <Picture 1>, the
    last-frame input becomes the next picture, the reference batch fills the
    slots after that, and each separately wired picture follows in the order
    it was given. A slot that is empty, because nothing is wired to it or
    because its source node is bypassed, is skipped rather than counted.
    """
    output = []
    for tensor in (first_frame, last_frame):
        if tensor is None:
            continue
        output.extend(tensor_to_pil_images(tensor, 1))
    if reference_images is not None and len(output) < maximum:
        output.extend(tensor_to_pil_images(reference_images, maximum - len(output)))
    for tensor in extra_images:
        if tensor is None or len(output) >= maximum:
            continue
        output.extend(tensor_to_pil_images(tensor, 1))
    return output[:maximum]
