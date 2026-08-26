import importlib.util
from pathlib import Path
import sys
import unittest

from h3_modes import (
    H3_MAX_CHARS,
    alignment_line,
    build_system_prompt,
    build_user_message,
    compile_prompt,
    detect_task,
    field_schema,
    resolve_task,
)
from media import _sample_indices, build_messages
from model_config import MODEL_IDS, resolve_model_id


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


class H3ModeTests(unittest.TestCase):
    def test_qwen_38_preset_is_available(self):
        self.assertEqual(MODEL_IDS["Qwen 3.8 27B"], "Qwen/Qwen3.8-27B")
        self.assertEqual(MODEL_IDS["Qwen 3.8 27B FP8"], "Qwen/Qwen3.8-27B-FP8")

    def test_task_detection_follows_reference_count(self):
        self.assertEqual(detect_task(0, False, False), "T2V")
        self.assertEqual(detect_task(1, False, False), "I2V")
        self.assertEqual(detect_task(1, False, True), "I2VA")
        self.assertEqual(detect_task(2, False, False), "FL2V")
        self.assertEqual(detect_task(3, False, False), "R2V")
        self.assertEqual(detect_task(3, False, True), "R2VA")
        self.assertEqual(detect_task(1, True, False), "V2V")
        self.assertEqual(detect_task(1, True, True), "V2VA")

    def test_explicit_task_beats_detection(self):
        self.assertEqual(resolve_task("R2V", 1, False, False), "R2V")
        with self.assertRaises(ValueError):
            resolve_task("X2V", 1, False, False)

    def test_i2v_alignment_line_is_fixed(self):
        line = alignment_line("I2V", 1, 5.0, False)
        self.assertIn("at 0.00 seconds into the target video", line)
        self.assertIn("<Picture 1> (from [Shot 1]) is fully referenced", line)

    def test_fl2v_alignment_line_carries_both_marks(self):
        line = alignment_line("FL2V", 2, 8.0, False)
        self.assertIn("0.00-second mark", line)
        self.assertIn("8.00-second mark", line)
        self.assertIn("<Picture 2>", line)

    def test_reference_alignment_is_opt_in(self):
        self.assertEqual(alignment_line("R2V", 3, 5.0, False), "")
        self.assertIn("<Picture 1>", alignment_line("R2V", 3, 5.0, True))

    def test_system_prompts_carry_no_worked_example(self):
        for task in ("T2V", "I2V", "FL2V", "R2V", "V2V", "I2VA", "R2VA", "V2VA"):
            text = build_system_prompt(task, 5.0, 2, 1, True)
            lowered = text.lower()
            self.assertNotIn("for example", lowered, task)
            self.assertNotIn("e.g.", lowered, task)
            self.assertNotIn("[shot 1] live", lowered, task)
            for field in field_schema(task):
                self.assertIn(field, text, f"{task}:{field}")

    def test_fl2v_prompt_explains_both_slots(self):
        text = build_system_prompt("FL2V", 5.0, 2, 0, False)
        self.assertIn("<Picture 1> is the exact first frame", text)
        self.assertIn("<Picture 2> is the exact last frame", text)

    def test_i2v_prompt_forbids_redescribing_fixed_appearance(self):
        text = build_system_prompt("I2V", 5.0, 1, 0, False)
        self.assertIn("Appearance is already fixed by the picture", text)

    def test_length_budget_scales_with_duration(self):
        short = build_system_prompt("T2V", 2.0, 0, 0, False)
        long = build_system_prompt("T2V", 15.0, 0, 0, False)
        self.assertIn("about 40 words", short)
        self.assertIn("about 210 words", long)

    def test_picture_slots_are_named_grammatically(self):
        one = build_system_prompt("I2V", 5.0, 1, 0, False)
        self.assertIn("One picture is attached. It is <Picture 1>.", one)
        three = build_system_prompt("R2V", 5.0, 3, 0, False)
        self.assertIn("3 pictures are attached", three)
        self.assertIn("<Picture 3>", three)

    def test_user_message_lists_available_tags(self):
        message = build_user_message("a cat", "R2V", 2, 1, True)
        self.assertIn("IDEA: a cat", message)
        self.assertIn("<Picture 1>", message)
        self.assertIn("<Picture 2>", message)
        self.assertIn("<Video 1>", message)
        self.assertIn("<Audio 1>", message)

    def test_compile_repairs_loose_model_output(self):
        raw = (
            "```\n"
            "Here is the prompt:\n"
            "Description: A quiet room.\n"
            "It holds still.\n"
            "Audio: room tone, a clock\n"
            "Music: N/A\n"
            "```"
        )
        compiled = compile_prompt(raw, "ALIGN LINE")
        self.assertTrue(compiled.startswith("ALIGN LINE"))
        self.assertIn("integrated_multimodal_description: [Shot 1]", compiled)
        self.assertIn("overall_soundscape: room tone, a clock", compiled)
        self.assertIn("non_diegetic_music: N/A", compiled)
        self.assertNotIn("```", compiled)
        self.assertEqual(compiled.count("\n\n"), 3)

    def test_compile_keeps_a_well_formed_answer(self):
        raw = (
            "integrated_multimodal_description: [Shot 1] A room.\n\n"
            "overall_soundscape: room tone.\n\n"
            "non_diegetic_music: N/A"
        )
        self.assertEqual(compile_prompt(raw, ""), raw)

    def test_reference_tasks_use_the_longer_field_set(self):
        self.assertEqual(field_schema("I2V")[0], "integrated_multimodal_description")
        schema = field_schema("R2V")
        self.assertEqual(schema[0], "subject_definitions")
        self.assertIn("retention_analysis", schema)
        self.assertIn("detailed_description", schema)

    def test_reference_prompt_asks_for_its_own_fields(self):
        text = build_system_prompt("R2V", 5.0, 3, 0, False)
        for field in field_schema("R2V"):
            self.assertIn(field, text, field)
        self.assertNotIn("integrated_multimodal_description", text)

    def test_compile_uses_the_reference_schema(self):
        raw = (
            "subject_definitions: <Picture 1> is the woman.\n"
            "summary: She waits.\n"
            "retention_analysis: Same face and coat.\n"
            "detailed_description: She waits by a door.\n"
            "overall_soundscape: room tone\n"
            "non_diegetic_music: N/A"
        )
        compiled = compile_prompt(raw, "", field_schema("R2V"))
        self.assertTrue(compiled.startswith("subject_definitions:"))
        self.assertIn("detailed_description: [Shot 1] She waits by a door.", compiled)
        self.assertIn("retention_analysis: Same face and coat.", compiled)

    def test_compile_recovers_a_mismatched_timeline_field(self):
        raw = "integrated_multimodal_description: [Shot 1] She waits.\n\nnon_diegetic_music: N/A"
        compiled = compile_prompt(raw, "", field_schema("R2V"))
        self.assertIn("detailed_description: [Shot 1] She waits.", compiled)

    def test_compile_caps_length(self):
        raw = "integrated_multimodal_description: [Shot 1] " + ("word " * 5000)
        self.assertLessEqual(len(compile_prompt(raw, "")), H3_MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
