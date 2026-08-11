import importlib.util
from pathlib import Path
import sys
import unittest

from media import _sample_indices, build_messages
from model_config import resolve_model_id


class CoreTests(unittest.TestCase):
    def test_comfy_package_entrypoint_loads(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "comfyui_vlm_prompt_enhancer",
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.assertIn("VLMPromptEnhancer", module.NODE_CLASS_MAPPINGS)

    def test_uniform_sampling_includes_ends(self):
        self.assertEqual(_sample_indices(10, 4), [0, 3, 6, 9])

    def test_custom_model_overrides_dropdown(self):
        self.assertEqual(resolve_model_id("Qwen 3.5 4B", "owner/custom-vlm"), "owner/custom-vlm")

    def test_messages_keep_system_and_references(self):
        marker = object()
        messages = build_messages("system", "improve me", [marker], None)
        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        self.assertEqual(messages[1]["content"][0], {"type": "image", "image": marker})
        self.assertEqual(messages[1]["content"][-1]["text"], "improve me")

    def test_invalid_custom_model_id_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_model_id("Qwen 3.5 4B", "not-a-repo")


if __name__ == "__main__":
    unittest.main()
