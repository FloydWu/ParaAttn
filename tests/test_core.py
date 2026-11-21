from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paraattn.attention import aligned_token_js, candidate_heads, choose_heads, js_divergence, normalize_mass, region_distribution
from paraattn.config import Config, config_from_dict
from paraattn.construction import arc_question, build_split
from paraattn.experiments import experiment_variants, summarize_runs
from paraattn.io import read_jsonl, seed_everything
from paraattn.modeling import attach_lora, forward_prefix
from paraattn.objective import training_objective
from paraattn.prompting import PromptEncoder, PromptError, make_prompt
from paraattn.schema import Question, check_disjoint
from paraattn.selection import select_heads, validate_selection_report
from paraattn.smoke import synthetic_splits, tiny_model, tiny_tokenizer
from paraattn.statistics import exact_mcnemar, metric_report, paired_comparison, wilson_interval
from paraattn.validation import generation_prompt, rule_failures, validate_candidate


torch.set_num_threads(min(2, torch.get_num_threads()))


def configuration():
    cfg = Config()
    cfg.model.name = "random-tiny-qwen2"
    cfg.model.device = "cpu"
    cfg.model.load_in_4bit = False
    cfg.model.lora_r = 2
    cfg.model.lora_alpha = 4
    cfg.model.lora_dropout = 0
    cfg.heads.layers = [1]
    cfg.heads.subset_size = 3
    cfg.heads.ratio = 0.5
    cfg.evaluation.diagnostic_layers = [1]
    cfg.evaluation.bootstrap_samples = 100
    return cfg.validate()


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.encoder = PromptEncoder(tiny_tokenizer())
        self.question = synthetic_splits()["train"][0]

    def test_template_and_spans(self):
        prompt, spans = make_prompt("Which gas?", ["air", "ice", "water", "rock"])
        self.assertEqual(prompt, "Choose the correct answer. Question: Which gas?\nA. air\nB. ice\nC. water\nD. rock\nAnswer:")
        self.assertEqual([prompt[a:b] for a, b in spans], ["Which gas?", "air", "ice", "water", "rock"])

    def test_content_regions_exclude_markers(self):
        encoded = self.encoder.encode(self.question.stem, self.question.options)
        self.assertEqual(set(encoded.region_ids), {-1, 0, 1, 2, 3, 4})
        self.assertEqual(encoded.region_ids[-1], -1)
        self.assertEqual(encoded.region_ids[-7:], [-1] * 7)

    def test_right_padding_last_real_query(self):
        encoded = [self.encoder.encode("Why?", self.question.options), self.encoder.encode("Why is snow white?", self.question.options)]
        batch = self.encoder.batch(encoded)
        self.assertEqual(batch["query_positions"].tolist(), [len(row.input_ids) - 1 for row in encoded])
        self.assertTrue((batch["region_ids"][0, len(encoded[0].input_ids):] == -1).all())

    def test_no_silent_truncation(self):
        with self.assertRaises(PromptError):
            PromptEncoder(tiny_tokenizer(), max_length=20).encode(self.question.stem, self.question.options)


class AttentionTests(unittest.TestCase):
    def test_sum_mask_and_normalization(self):
        rows = torch.tensor([[[100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0]]], requires_grad=True)
        ids = torch.tensor([[-1, 0, 1, 2, 3, 4, -1]])
        distribution = region_distribution(rows, ids)
        torch.testing.assert_close(distribution, torch.tensor([[[.10, .15, .20, .25, .30]]]))
        distribution[0, 0, 0].backward()
        self.assertEqual(rows.grad[0, 0, 0], 0)
        self.assertEqual(rows.grad[0, 0, -1], 0)
        self.assertNotEqual(rows.grad[0, 0, 1], 0)

    def test_zero_mass_uniform_smoothing(self):
        distribution = normalize_mass(torch.zeros(2, 5))
        torch.testing.assert_close(distribution, torch.full((2, 5), .2))

    def test_js_identity_symmetry_and_bound(self):
        p = torch.tensor([[1.0, 0.0], [.1, .9]])
        q = torch.tensor([[0.0, 1.0], [.8, .2]])
        torch.testing.assert_close(js_divergence(p, p), torch.zeros(2))
        torch.testing.assert_close(js_divergence(p, q), js_divergence(q, p))
        self.assertAlmostEqual(js_divergence(p, q)[0].item(), math.log(2), places=6)

    def test_paper_count_and_tie_break(self):
        heads = candidate_heads(list(range(16, 24)), 24, 14)
        self.assertEqual(len(heads), 112)
        scores = [{"layer": l, "head": h, "mean_js": 1.0} for l, h in reversed(heads)]
        chosen = choose_heads(scores, .2, "sensitive", 42)
        self.assertEqual(len(chosen), 22)
        self.assertEqual(chosen, heads[:22])

    def test_all_low_and_random(self):
        scores = [{"layer": 1, "head": h, "mean_js": float(h)} for h in range(4)]
        self.assertEqual(choose_heads(scores, .5, "low", 42), [(1, 0), (1, 1)])
        self.assertEqual(len(choose_heads(scores, .5, "all", 42)), 4)
        self.assertEqual(choose_heads(scores, .5, "random", 42), choose_heads(scores, .5, "random", 42))
        with self.assertRaises(ValueError):
            choose_heads(scores, .01, "sensitive", 42)


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.cfg = configuration()
        self.tokenizer = tiny_tokenizer()
        self.encoder = PromptEncoder(self.tokenizer)
        self.model = tiny_model(self.tokenizer).eval()
        self.questions = synthetic_splits()["train"][:2]

    def test_padded_logits_equal_unpadded_logits(self):
        encoded = [self.encoder.encode("Why?", self.questions[0].options),
                   self.encoder.encode("Why is snow white?", self.questions[0].options)]
        batched = forward_prefix(self.model, self.encoder.batch(encoded), [(1, 0)])
        single = forward_prefix(self.model, self.encoder.batch([encoded[0]]), [(1, 0)])
        torch.testing.assert_close(batched.logits[0], single.logits[0], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(batched.rows[0, :, :len(encoded[0].input_ids)], single.rows[0], atol=2e-6, rtol=1e-5)
        self.assertEqual(float(batched.rows[0, :, len(encoded[0].input_ids):].sum().detach()), 0)

    def test_selected_query_head_is_correct_under_gqa(self):
        encoded = [self.encoder.encode(self.questions[0].stem, self.questions[0].options)]
        batch = self.encoder.batch(encoded)
        selected = forward_prefix(self.model, batch, [(0, 3), (1, 1)])
        full = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          output_attentions=True, use_cache=False)
        self.assertEqual(full.attentions[0].shape[1], 4)
        torch.testing.assert_close(selected.rows[:, 0], full.attentions[0][:, 3, -1])

    def test_final_query_cannot_attend_future_gold_label(self):
        encoded = self.encoder.encode(self.questions[0].stem, self.questions[0].options)
        batch = self.encoder.batch([encoded])
        before = forward_prefix(self.model, batch, [(1, 0)])
        extended = torch.cat([batch["input_ids"], torch.tensor([[self.encoder.label_token_ids[0]]])], dim=1)
        full = self.model(input_ids=extended, attention_mask=torch.ones_like(extended), output_attentions=True, use_cache=False)
        q = len(encoded.input_ids) - 1
        torch.testing.assert_close(before.rows[0, 0], full.attentions[1][0, 0, q, :q + 1], atol=1e-6, rtol=1e-5)
        self.assertEqual(full.attentions[1][0, 0, q, q + 1].item(), 0)

    def test_region_regularizer_reaches_lora_query_parameters(self):
        self.model = attach_lora(self.model, self.cfg.model).eval()
        original = [self.encoder.encode(q.stem, q.options) for q in self.questions]
        paraphrase = [self.encoder.encode(q.paraphrases[0], q.options) for q in self.questions]
        batch = self.encoder.batch(original + paraphrase)
        result = forward_prefix(self.model, batch, [(1, 0), (1, 2)])
        distributions = region_distribution(result.rows, batch["region_ids"])
        regularizer = js_divergence(distributions[:2], distributions[2:]).mean()
        regularizer.backward()
        gradient = sum(float(p.grad.abs().sum()) for name, p in self.model.named_parameters()
                       if "q_proj.lora_B" in name and p.grad is not None)
        self.assertGreater(gradient, 0)
        self.assertFalse(any(p.requires_grad for name, p in self.model.named_parameters() if "lora_" not in name))

    def test_all_training_methods_have_finite_gradients(self):
        self.model = attach_lora(self.model, self.cfg.model).eval()
        for method in ["standard_lora", "augmentation", "output_js", "paraattn"]:
            self.model.zero_grad()
            self.cfg.training.method = method
            loss, _ = training_objective(self.model, self.encoder, self.questions, self.cfg, [(1, 0)])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_paired_task_losses_are_summed(self):
        self.model = attach_lora(self.model, self.cfg.model).eval()
        self.cfg.training.method = "augmentation"
        loss, components = training_objective(self.model, self.encoder, self.questions, self.cfg)
        self.assertAlmostEqual(loss.item(), components["ce_original"] + components["ce_paraphrase"], places=5)

    def test_token_alignment_ablations_keep_gradients(self):
        self.model = attach_lora(self.model, self.cfg.model).eval()
        for representation in ["position", "matching"]:
            self.model.zero_grad()
            self.cfg.training.representation = representation
            loss, components = training_objective(self.model, self.encoder, self.questions, self.cfg, [(1, 0)])
            self.assertTrue(math.isfinite(components["consistency_loss"]))
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in self.model.parameters() if p.requires_grad))

    def test_selection_only_uses_train_and_is_reproducible(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        questions = synthetic_splits()["train"]
        a = select_heads(self.model, self.encoder, questions, self.cfg)
        b = select_heads(self.model, self.encoder, questions, self.cfg)
        self.assertEqual(a, b)
        self.assertTrue(all("train" in qid for qid in a["selection_subset_ids"]))
        self.assertEqual(len(validate_selection_report(a, self.model, questions, self.cfg)), 2)
        changed = copy.deepcopy(questions); changed[0].paraphrases = ["A changed stem?"]
        with self.assertRaises(ValueError):
            validate_selection_report(a, self.model, changed, self.cfg)


class DataTests(unittest.TestCase):
    def test_arc_order_and_gold_preservation(self):
        row = {"id": "q", "question": " Which color? ", "choices": {"text": ["red", "blue", "white", "black"], "label": list("ABCD")}, "answerKey": "C"}
        q = arc_question(row, "train")
        self.assertEqual(q.label, 2)
        self.assertEqual(q.options, row["choices"]["text"])
        row["choices"]["label"] = list("1234")
        with self.assertRaises(ValueError):
            arc_question(row, "train")

    def test_cross_split_duplicate_detected_despite_new_id(self):
        q = synthetic_splits()["train"][0]
        duplicate = copy.deepcopy(q); duplicate.id = "different-id"
        with self.assertRaises(ValueError):
            check_disjoint({"train": [q], "test": [duplicate]})

    def test_rules_reject_numbers_units_negation_and_cues(self):
        failures = rule_failures("Which is not 2 cm long?", "The correct answer is 3 m long.", ["a", "b", "c", "d"], [])
        self.assertTrue({"answer_cue", "numbers_changed", "units_changed", "negation_changed"} <= set(failures))

    def test_list_format_is_rejected_before_whitespace_normalization(self):
        q = synthetic_splits()["train"][0]
        result = validate_candidate(q, "Which color?\nA. white\nB. red", [], lambda a, b: 1, lambda p: "PASS", .85)
        self.assertIn("not_single_stem", result.failures)

    def test_similarity_and_verifier_are_required(self):
        q = synthetic_splits()["train"][0]
        result = validate_candidate(q, q.paraphrases[0], [], lambda a, b: .84, lambda p: "PASS", .85)
        self.assertFalse(result.accepted)
        result = validate_candidate(q, q.paraphrases[0], [], lambda a, b: .9, lambda p: "FAIL", .85)
        self.assertFalse(result.accepted)
        result = validate_candidate(q, q.paraphrases[0], [], lambda a, b: .9, lambda p: "PASS", .85)
        self.assertTrue(result.accepted)

    def test_greedy_slots_use_different_prompts(self):
        q = synthetic_splits()["train"][0]
        self.assertNotEqual(generation_prompt(q, 0, [], []), generation_prompt(q, 1, [], []))
        self.assertNotEqual(generation_prompt(q, 0, [], []), generation_prompt(q, 0, [], ["duplicate"]))

    def test_attempt_logging_retries_and_two_test_paraphrases(self):
        cfg = configuration()
        q = synthetic_splits()["test"][0]
        q.paraphrases = []
        candidates = iter([q.stem, f"In sample test 0, which color describes snow?", f"What is the color of snow for sample test 0?"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "attempts.jsonl"
            accepted = build_split([q], 1, 2, lambda p: next(candidates), lambda a, b: .99,
                                   lambda p: "PASS", cfg, PromptEncoder(tiny_tokenizer()), path, "test")
            logs = read_jsonl(path)
        self.assertEqual(len(accepted[0].paraphrases), 2)
        self.assertEqual(len(logs), 3)
        self.assertFalse(logs[0]["validation"]["accepted"])
        self.assertIn("GOLD LABEL: A", logs[1]["verifier_prompt"])
        self.assertEqual(accepted[0].options, q.options)
        self.assertEqual(accepted[0].label, q.label)


def predictions():
    rows = []
    for i, (original, paraphrases) in enumerate([(0, [0, 0]), (1, [1, 1]), (0, [0, 1]), (0, [0, 0])]):
        rows.append({"id": str(i), "label": 0, "group_hash": "hash" + str(i),
                     "original": {"prediction": original}, "paraphrases": [{"prediction": p} for p in paraphrases]})
    return rows


class StatisticsTests(unittest.TestCase):
    def test_group_metrics_distinguish_consistency_from_correctness(self):
        report = metric_report(predictions(), 100, 42)
        self.assertEqual(report["metrics"], {"original_accuracy": .75, "paraphrase_accuracy": .625,
                                             "consistency": .75, "worst_case_accuracy": .5})
        self.assertEqual(report["paraphrase_realizations"], 8)

    def test_reported_wilson_values(self):
        lo, hi = wilson_interval(76, 100)
        self.assertAlmostEqual(lo, .668, places=3)
        self.assertAlmostEqual(hi, .833, places=3)
        lo, hi = wilson_interval(151, 200)
        self.assertAlmostEqual(lo, .691, places=3)
        self.assertAlmostEqual(hi, .809, places=3)

    def test_exact_mcnemar(self):
        result = exact_mcnemar(np.ones(10), np.zeros(10))
        self.assertAlmostEqual(result["two_sided_p"], 2 / 1024)
        self.assertEqual(exact_mcnemar([1, 0], [1, 0])["two_sided_p"], 1)

    def test_paired_same_predictions_zero_difference(self):
        result = paired_comparison(predictions(), list(reversed(predictions())), 100)
        self.assertTrue(all(v == 0 for v in result["metric_difference"].values()))
        self.assertTrue(all(v == [0, 0] for v in result["paired_cluster_bootstrap_95_ci"].values()))

    def test_paired_mismatched_groups_fail(self):
        right = predictions(); right[0]["group_hash"] = "changed"
        with self.assertRaises(ValueError):
            paired_comparison(predictions(), right, 100)

    def test_seed_aggregation_does_not_invent_std_for_single_run(self):
        rows = [{"variant": "paraattn", "seed": 42, "report": {"test_data_hash": "same", "metrics": {"original_accuracy": .7}}}]
        result = summarize_runs(rows)
        self.assertIsNone(result["variants"]["paraattn"]["metrics"]["original_accuracy"]["sample_std"])
        rows.append({"variant": "paraattn", "seed": 43, "report": {"test_data_hash": "same", "metrics": {"original_accuracy": .9}}})
        self.assertAlmostEqual(summarize_runs(rows)["variants"]["paraattn"]["metrics"]["original_accuracy"]["mean"], .8)


class ConfigTests(unittest.TestCase):
    def test_config_roundtrip_and_unknown_keys(self):
        self.assertEqual(config_from_dict(Config().to_dict()).to_dict(), Config().to_dict())
        with self.assertRaises(ValueError):
            config_from_dict({"training": {"batch_szie": 4}})

    def test_suites_cover_manuscript_tables(self):
        cfg = Config()
        counts = {suite: len(list(experiment_variants(cfg, suite))) for suite in ["baselines", "heads", "representations", "layers", "weights"]}
        self.assertEqual(counts, {"baselines": 4, "heads": 5, "representations": 4, "layers": 4, "weights": 6})


if __name__ == "__main__":
    unittest.main()
