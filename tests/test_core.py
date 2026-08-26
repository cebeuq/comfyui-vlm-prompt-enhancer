import importlib.util
from pathlib import Path
import sys
import unittest

from h3_modes import (
    H3_MAX_CHARS,
    alignment_line,
    build_system_prompt,
    build_user_message,
    check_image_count,
    compile_prompt,
    detect_task,
    field_schema,
    resolve_ref_kinds,
    resolve_task,
    scrub_hedges,
    snap_seconds,
    strip_reasoning,
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
        self.assertEqual(detect_task(0, False, False), "T2VA")
        self.assertEqual(detect_task(1, False, False), "I2VA")
        self.assertEqual(detect_task(2, False, False), "FL2VA")
        self.assertEqual(detect_task(3, False, False), "Ref2VA")
        self.assertEqual(detect_task(1, False, True), "Ref2VA")
        self.assertEqual(detect_task(1, True, False), "Ref2VA")

    def test_explicit_task_beats_detection(self):
        self.assertEqual(resolve_task("L2VA", 1, False, False), "L2VA")
        with self.assertRaises(ValueError):
            resolve_task("X2V", 1, False, False)

    def test_old_task_codes_still_load(self):
        self.assertEqual(resolve_task("I2V", 1, False, False), "I2VA")
        self.assertEqual(resolve_task("FL2V", 2, False, False), "FL2VA")
        self.assertEqual(resolve_task("R2V", 3, False, False), "Ref2VA")
        self.assertEqual(resolve_task("V2VA", 1, True, True), "Ref2VA")

    def test_i2va_alignment_line_is_fixed(self):
        line = alignment_line("I2VA", 1, 5, False)
        self.assertEqual(
            line,
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.",
        )

    def test_fl2va_alignment_line_uses_the_official_bare_wording(self):
        line = alignment_line("FL2VA", 2, 8, False)
        self.assertEqual(
            line,
            "How the reference pictures align with the target video - "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            "Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.",
        )

    def test_l2va_alignment_line_marks_the_end(self):
        line = alignment_line("L2VA", 1, 10, False)
        self.assertIn("10.00-second mark", line)
        self.assertIn("<Picture 1> (from [Shot 1])", line)

    def test_reference_alignment_is_opt_in(self):
        self.assertEqual(alignment_line("Ref2VA", 3, 5, False), "")
        self.assertIn("<Picture 1>", alignment_line("Ref2VA", 3, 5, True))

    def test_duration_snaps_into_the_supported_range(self):
        self.assertEqual(snap_seconds(1), 4)
        self.assertEqual(snap_seconds(5.4), 5)
        self.assertEqual(snap_seconds(99), 15)

    def test_image_count_is_enforced_per_task(self):
        check_image_count("FL2VA", 2)
        check_image_count("Ref2VA", 9)
        with self.assertRaises(ValueError):
            check_image_count("I2VA", 2)
        with self.assertRaises(ValueError):
            check_image_count("T2VA", 1)

    def test_reference_kind_label_is_derived(self):
        self.assertEqual(resolve_ref_kinds("Auto", 2, False, False, False), "reference generation")
        self.assertEqual(
            resolve_ref_kinds("Auto", 2, False, True, True),
            "reference generation + keyframe completion + audio reference",
        )
        self.assertEqual(resolve_ref_kinds("video editing", 2, True, False, False), "video editing")

    def test_leaked_reasoning_is_removed(self):
        self.assertEqual(strip_reasoning("<think>plan</think>answer"), "answer")
        self.assertEqual(strip_reasoning("noise</think>answer"), "answer")

    def test_official_shake_term_survives_the_hedge_scrub(self):
        self.assertIn("Shake Slightly", scrub_hedges("The camera does Shake Slightly here."))
        self.assertNotIn("slightly", scrub_hedges("She slightly turns."))

    def test_system_prompts_carry_no_worked_example(self):
        for task in ("T2VA", "I2VA", "L2VA", "FL2VA", "Ref2VA"):
            text = build_system_prompt(task, 5.0, 2, 1, True)
            lowered = text.lower()
            self.assertNotIn("for example", lowered, task)
            self.assertNotIn("e.g.", lowered, task)
            self.assertNotIn("[shot 1] live", lowered, task)
            for field in field_schema(task):
                self.assertIn(field, text, f"{task}:{field}")

    def test_fl2va_prompt_explains_both_slots(self):
        text = build_system_prompt("FL2VA", 5, 2)
        self.assertIn("<Picture 1> is the exact first frame", text)
        self.assertIn("<Picture 2> is the exact last frame", text)

    def test_l2va_prompt_works_backwards(self):
        text = build_system_prompt("L2VA", 5, 1)
        self.assertIn("<Picture 1> is the exact last frame", text)
        self.assertIn("Work backwards", text)

    def test_i2va_prompt_forbids_redescribing_fixed_appearance(self):
        text = build_system_prompt("I2VA", 5, 1)
        self.assertIn("Appearance is already fixed", text)

    def test_frame_modes_stay_single_shot_and_reference_modes_do_not(self):
        self.assertIn("holds [Shot 1] and nothing else", build_system_prompt("FL2VA", 5, 2))
        self.assertIn("Write a second shot only when", build_system_prompt("Ref2VA", 5, 2))

    def test_camera_vocabulary_is_the_official_table(self):
        text = build_system_prompt("T2VA", 5, 0)
        for term in ("Pedestal Up", "Arc Shot", "Roll Clockwise", "Shake Slightly", "Static Shot"):
            self.assertIn(term, text, term)

    def test_reference_prompt_carries_the_retention_markers(self):
        text = build_system_prompt("Ref2VA", 5, 2, ref_kinds="reference generation")
        for marker in ("fully_preserved", "attribute_transfer", "fully_copy", "weak_reference"):
            self.assertIn(marker, text, marker)
        self.assertIn("[reference generation]", text)
        self.assertIn("<Subject 1>", text)

    def test_length_budget_scales_with_duration(self):
        short = build_system_prompt("T2VA", 4, 0)
        long = build_system_prompt("T2VA", 15, 0)
        self.assertIn("about 65 words", short)
        self.assertIn("about 210 words", long)
        self.assertIn("about 350 words", build_system_prompt("Ref2VA", 5, 2))

    def test_picture_slots_are_named_grammatically(self):
        one = build_system_prompt("I2VA", 5, 1)
        self.assertIn("One picture is attached. It is <Picture 1>.", one)
        three = build_system_prompt("Ref2VA", 5, 3)
        self.assertIn("3 pictures are attached", three)
        self.assertIn("<Picture 3>", three)

    def test_user_message_lists_available_tags(self):
        message = build_user_message("a cat", "Ref2VA", 2, 1, 1)
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
        self.assertEqual(field_schema("I2VA")[0], "integrated_multimodal_description")
        schema = field_schema("Ref2VA")
        self.assertEqual(schema[0], "subject_definitions")
        self.assertIn("retention_analysis", schema)
        self.assertIn("detailed_description", schema)

    def test_reference_prompt_asks_for_its_own_fields(self):
        text = build_system_prompt("Ref2VA", 5, 3)
        for field in field_schema("Ref2VA"):
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
        compiled = compile_prompt(raw, "", field_schema("Ref2VA"))
        self.assertTrue(compiled.startswith("subject_definitions:"))
        self.assertIn("detailed_description: [Shot 1] She waits by a door.", compiled)
        self.assertIn("retention_analysis: Same face and coat.", compiled)

    def test_compile_recovers_a_mismatched_timeline_field(self):
        raw = "integrated_multimodal_description: [Shot 1] She waits.\n\nnon_diegetic_music: N/A"
        compiled = compile_prompt(raw, "", field_schema("Ref2VA"))
        self.assertIn("detailed_description: [Shot 1] She waits.", compiled)

    def test_compile_strips_hedging_adverbs(self):
        raw = "integrated_multimodal_description: [Shot 1] Her shoulders slightly shift. The light gently falls."
        compiled = compile_prompt(raw, "")
        self.assertIn("Her shoulders shift.", compiled)
        self.assertIn("The light falls.", compiled)
        self.assertNotIn("slightly", compiled)
        self.assertNotIn("gently", compiled)

    def test_style_rule_names_a_closed_vocabulary(self):
        text = build_system_prompt("I2VA", 5, 1)
        self.assertIn("live-action cinematic, documentary, 3D render", text)

    def test_compile_caps_length_without_losing_the_sound_fields(self):
        raw = (
            "integrated_multimodal_description: [Shot 1] " + ("word " * 5000) + "\n\n"
            "overall_soundscape: room tone\n\n"
            "non_diegetic_music: N/A"
        )
        compiled = compile_prompt(raw, "")
        self.assertLessEqual(len(compiled), H3_MAX_CHARS)
        self.assertIn("overall_soundscape: room tone", compiled)
        self.assertTrue(compiled.rstrip().endswith("non_diegetic_music: N/A"))


if __name__ == "__main__":
    unittest.main()
