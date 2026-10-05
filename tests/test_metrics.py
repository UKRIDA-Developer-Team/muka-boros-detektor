import importlib
import importlib.util
import unittest


class MetricsTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('faceage'), 'Metrik belum dibuat')
        return importlib.import_module('faceage.metrics')

    def test_known_regression_errors_and_age_groups(self):
        metrics = self.module()
        result = metrics.regression_metrics([10,20,30], [12,18,40])
        self.assertAlmostEqual(result['mae'], 14/3)
        self.assertEqual(result['rmse'], 6.0)
        self.assertEqual(result['median_absolute_error'], 2.0)
        per_group = metrics.age_group_metrics([10,20,30], [12,18,40])
        self.assertEqual(per_group['0-12']['count'], 1)
        self.assertEqual(per_group['0-12']['mae'], 2.0)
        self.assertEqual(per_group['20-39']['count'], 2)
        self.assertEqual(per_group['20-39']['mae'], 6.0)
        self.assertIsNone(per_group['60+']['mae'])

    def test_bad_predictions_cannot_produce_a_plausible_score(self):
        metrics = self.module()
        for targets, values in [([],[]),([1],[1,2]),([1],[float('nan')]),([1],[float('inf')])]:
            with self.assertRaises(ValueError): metrics.regression_metrics(targets,values)
