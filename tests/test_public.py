import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from verify_public import verify
from audit_private_artifacts import metrics, prepared_weighted_auc, weighted_auc


class PublicEvidenceTests(unittest.TestCase):
    def test_original_bytes_notebook_syntax_summary_and_model_export(self):
        self.assertEqual(verify()["inherited_files"], 233)

    def test_dkt_observed_auc_higher_each_seed(self):
        data = pd.read_csv(ROOT / "results/hyperkt_stage10/per_seed_test_metrics.csv")
        wide = data.pivot(index="seed", columns="variant", values="auc")
        self.assertTrue((wide.DKT_matched > wide.HyperKT_matched).all())
        self.assertTrue((wide.DKT_matched > wide.HyperKT_gated).all())

    def test_paired_intervals_contain_zero_in_four_original_comparisons(self):
        data = pd.read_csv(ROOT / "results/hyperkt_stage10/paired_auc_comparisons.csv")
        original = data[data.comparator == "HyperKT_matched"]
        self.assertEqual(((original.student_boot_ci_low <= 0) & (original.student_boot_ci_high >= 0)).sum(), 4)

    def test_metrics_tied_auc(self):
        self.assertEqual(metrics([0, 1, 0, 1], [.5, .5, .5, .5])["auc"], .5)

    def test_weighted_auc_equals_repeated_learner_rows(self):
        y = np.array([0, 1, 1, 0, 0, 1])
        p = np.array([.2, .8, .5, .5, .3, .9])
        cluster = np.array([0, 0, 1, 1, 2, 2])
        multiplicity = np.array([2, 0, 3])
        idx = np.repeat(np.arange(len(y)), multiplicity[cluster])
        self.assertAlmostEqual(weighted_auc(prepared_weighted_auc(y, p, cluster), multiplicity), metrics(y[idx], p[idx])["auc"])

    def test_private_audit_reports_all_three_splits(self):
        report = json.loads((ROOT / "analysis/private-artifact-audit.json").read_text())
        self.assertEqual(set(report["sequences"]["splits"]), {"train", "val", "test"})
        self.assertEqual(report["test_results"]["aligned_predictions_each"], 32522)
        self.assertEqual(len(report["test_results"]["individual_auc_intervals"]), 15)
        self.assertEqual(len(report["test_results"]["paired_auc_intervals"]), 10)

    def test_no_private_file_types_in_public_tree(self):
        blocked = (".pt", ".parquet", ".zip")
        for folder in ("results", "analysis", "provenance", "notebooks"):
            for path in (ROOT / folder).rglob("*"):
                if path.is_file():
                    self.assertFalse(path.name.endswith(blocked), path)
                    self.assertFalse(path.name.endswith("_sequences.npz"), path)
                    self.assertFalse(path.name.startswith("predictions_"), path)

    def test_final_checkpoint_verification_scope(self):
        report = json.loads((ROOT / "analysis/checkpoint-verification.json").read_text())
        self.assertEqual(report["count"], 15)
        self.assertFalse(report["training_performed"])
        self.assertTrue(report["frozen_test_inference_rerun"])
        self.assertTrue(all(row["strict_state_load"] and row["all_tensors_finite"] for row in report["checkpoints"]))
        self.assertTrue(all(row["causal_probe_max_delta"] < 1e-6 for row in report["checkpoints"]))

    def test_checkpoint_manifest_has_43_selected_states(self):
        report = json.loads((ROOT / "provenance/checkpoint-release-manifest.json").read_text())
        self.assertEqual(len(report["checkpoints"]), 43)
        self.assertEqual(sum(row["checkpoints"] for row in report["assets"]), 43)
        self.assertEqual(len(report["excluded_resume_states"]), 15)
        self.assertTrue(all(row["weights_only_load"] and row["finite_tensor_tree"] for row in report["checkpoints"]))

    def test_cpu_cuda_agreement_flags_are_not_silently_relaxed(self):
        rows = json.loads((ROOT / "analysis/checkpoint-verification.json").read_text())["checkpoints"]
        self.assertTrue(all(not row["probabilities_close_at_1e_5"] for row in rows if row["variant"] == "DKT_matched"))
        self.assertTrue(all(row["probabilities_close_at_1e_5"] for row in rows if row["variant"] != "DKT_matched"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch not installed; see reproduction guide")
class SyntheticModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.H = np.zeros((151, 16), dtype=np.float32)

    def test_all_final_variants_output_shape_and_finite(self):
        from hyperkt.loading import make_model
        for variant in ("DKT_matched", "HyperKT_matched", "HyperKT_gated"):
            model = make_model(variant, self.H).eval()
            with self.torch.no_grad():
                out = model(self.torch.full((2, 10), 2), self.torch.zeros((2, 10), dtype=self.torch.long), self.torch.full((2, 10), 3))
            self.assertEqual(tuple(out.shape), (2, 10))
            self.assertTrue(self.torch.isfinite(out).all())

    def test_future_responses_do_not_change_past_outputs(self):
        from hyperkt.loading import make_model
        for variant in ("DKT_matched", "HyperKT_matched", "HyperKT_gated"):
            model = make_model(variant, self.H).eval()
            hs = self.torch.full((2, 20), 2)
            ts = self.torch.full((2, 20), 3)
            hc = self.torch.zeros((2, 20), dtype=self.torch.long)
            changed = hc.clone()
            changed[:, 10:] = 1
            with self.torch.no_grad():
                self.assertTrue(self.torch.allclose(model(hs, hc, ts)[:, :10], model(hs, changed, ts)[:, :10], atol=1e-7, rtol=0))

    def test_hypergraph_assignments_are_buffers_not_trainable_parameters(self):
        from hyperkt.loading import make_model
        model = make_model("HyperKT_matched", self.H)
        self.assertIn("assignments", dict(model.named_buffers()))
        self.assertNotIn("assignments", dict(model.named_parameters()))
        self.assertIn("edge_embedding", dict(model.named_parameters()))

    def test_safe_loader_rejects_wrong_hash_before_deserializing(self):
        import tempfile
        from hyperkt.loading import load_checkpoint
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b"not a torch pickle")
            handle.flush()
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_checkpoint(handle.name, "DKT_matched", self.H, "0" * 64)


if __name__ == "__main__":
    unittest.main()
