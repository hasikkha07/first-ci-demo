
import os
import json
import unittest
import pandas as pd

class TestMushroomMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("mushrooms.csv"))

    def test_dataset_columns(self):
        data = pd.read_csv("mushrooms.csv")
        self.assertIn("class", data.columns)
        self.assertEqual(len(data), 8124)

    def test_model_exists(self):
        self.assertTrue(os.path.exists("mushroom_model.pkl"))

    def test_metrics_exist(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

if __name__ == "__main__":
    unittest.main()
