import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset import DATA, ROOT, load_dataset
from evaluation import confusion, scores
from features import HZ, LABELS, LENGTH, NAMES, extract
from inference import classify, load_model, probabilities, read_window
from verify import verify


class FeatureTests(unittest.TestCase):
    def test_constant_window_has_no_spurious_frequency(self):
        values = extract(np.ones((2, LENGTH, 3)))
        self.assertEqual(values.shape, (2, len(NAMES)))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[0, NAMES.index("x_sd_g")], 0)
        self.assertEqual(values[0, NAMES.index("x_peak_hz")], 0)

    def test_known_frequency_lands_in_expected_band(self):
        t = np.arange(LENGTH) / HZ
        window = np.column_stack([np.sin(2 * np.pi * 3.125 * t), np.zeros(LENGTH), np.ones(LENGTH)])
        features = extract(window[None])[0]
        self.assertAlmostEqual(features[NAMES.index("x_peak_hz")], 3.125)
        self.assertGreater(features[NAMES.index("x_power_2_5")], 0.99)

    def test_invalid_shape_or_nonfinite_rejected(self):
        for values in (np.zeros((1, 64, 3)), np.full((1, LENGTH, 3), np.nan)):
            with self.assertRaises(ValueError):
                extract(values)

    def test_bad_cadence_is_detected_before_scoring(self):
        text = (
            (DATA / "examples/activity-1.csv")
            .read_text(encoding="utf-8")
            .replace("\n20,", "\n35,", 1)
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.csv"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cadência"):
                read_window(path)


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()
        cls.model = load_model()
        cls.report = json.loads((ROOT / "release/evaluation.json").read_text(encoding="utf-8"))

    def test_official_split_and_oof_have_no_subject_leakage(self):
        train = set(self.ds["subject_train"])
        test = set(self.ds["subject_test"])
        held = [p for fold in self.report["cv_validation_subjects"] for p in fold]
        self.assertFalse(train & test)
        self.assertEqual((len(train), len(test)), (21, 9))
        self.assertEqual(sorted(held), sorted(train))
        self.assertEqual(sorted(test), self.report["test_subjects"])

    def test_examples_recreate_their_heldout_features(self):
        meta = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        for item in meta["examples"]:
            csvpath = DATA / "examples" / item["file"]
            idx = item["official_test_row_zero_based"]
            np.testing.assert_allclose(
                extract(read_window(csvpath)[None])[0], self.ds["x_test"][idx], atol=1e-6
            )
            self.assertEqual(item["label"], LABELS[int(self.ds["y_test"][idx]) - 1])

    def test_model_softmax_and_abstention(self):
        x = self.ds["x_test"][:20]
        scores_ = probabilities(x, self.model)
        self.assertTrue(np.all((scores_ >= 0) & (scores_ <= 1)))
        np.testing.assert_allclose(scores_.sum(axis=1), 1)
        model = dict(self.model, threshold=1.0)
        self.assertEqual(classify(DATA / "examples/activity-1.csv", model)["decisao"], "revisar")

    def test_every_released_prediction_replays(self):
        result = verify()
        self.assertEqual(result["windows"], 2947)
        self.assertAlmostEqual(result["coverage"], self.report["selective"]["test_coverage"])
        self.assertGreater(result["macro_f1"], self.report["baseline_most_frequent"]["macro_f1"])

    def test_confusion_counts_and_absent_class_convention(self):
        cm = confusion([1, 1, 2], [1, 2, 2])
        metrics = scores(cm)
        self.assertEqual(metrics["per_class"]["deitado"]["support"], 0)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 2 / 3) / 6)
