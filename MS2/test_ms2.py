# Test file for the MS2 project.
#
# Author(s):
# - CS-233 Team
# Version: 20250723

"""
Module for auto-testing student projects.
This is based on the file from Francois Fleuret's
"Deep Learning Course": https://fleuret.org/dlc/.

This is the Milestone 2 version.
"""

import re
import sys
import os
import unittest
import importlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class HidePrints:
    """Disable normal printing for calling student code."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class NoHidePrints:
    """Don't disable normal printing for calling student code."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

no_print = HidePrints
class TestProject(unittest.TestCase):

    @staticmethod
    def title(msg):
        print(f"\n==============\n> {msg} ...")

    def test_1_folder_structure(self):
        project_path = Path(".")
        """Test the framework structure (folder and files)."""
        self.title("Testing folder structure")
        self.assertTrue(project_path.exists(), f"No folder found at {project_path}")

        # Main files
        for file in ["main.py", "report.pdf"]:
            with self.subTest(f"Checking file {file}"):
                self.assertTrue((project_path / file).exists(), f"No file {file} found at {project_path}")

        # Source code
        src_path = project_path / "src"
        self.assertTrue(src_path.exists(), f"{src_path} not found")
        for file in ["__init__.py", "data.py", "utils.py"]:
            with self.subTest(f"Checking file src/{file}"):
                self.assertTrue((src_path / file).exists(), f"No file {file} found at {src_path}")
        # Methods
        method_path = src_path / "methods"
        self.assertTrue(method_path.exists(), f"{method_path} not found")
        for file in ["__init__.py", "dummy_methods.py",
                      "deep_network.py"]:
            with self.subTest(f"Checking file methods/{file}"):
                self.assertTrue((method_path / file).exists(), f"No file {file} found at {method_path}")


    def _import_and_test(self, name, class_name, *args, **kwargs):
        """Test the import of the method and its functions."""
        # Code structure
        module = importlib.import_module(f"src.methods.{name}")
        method = module.__getattribute__(class_name)(*args, **kwargs)
        for fn in ["fit", "predict"]:
            _ = method.__getattribute__(fn)

        # Functions inputs and outputs
        N, D = 10, 3
        training_data = np.random.rand(N, D)
        training_labels = np.random.randint(0, D, N)
        test_data = np.random.rand(N, D)
        with no_print():
            pred_labels = method.fit(training_data, training_labels)
        self.assertIsInstance(pred_labels, np.ndarray, f"{name}.{class_name}.fit() should output an array, not {type(pred_labels)}")
        self.assertEqual(pred_labels.shape, training_labels.shape, f"{name}.{class_name}.fit() output has wrong shape ({pred_labels.shape} != {training_labels.shape})")
        with no_print():
            pred_labels = method.predict(test_data)
        self.assertIsInstance(pred_labels, np.ndarray, f"{name}.{class_name}.predict() should output an array, not {type(pred_labels)}")
        self.assertEqual(pred_labels.shape, training_labels.shape, f"{name}.{class_name}.predict() output has wrong shape ({pred_labels.shape} != {training_labels.shape})")

        return method


    def test_2_dummy_methods(self):
        """Test the dummy methods."""
        self.title("Testing dummy methods")

        _ = self._import_and_test("dummy_methods", "DummyClassifier",
                                  arg1=1)


    def test_4b_deep_network(self):
        self.title("Testing deep-network")

        # For dummy data
        D, C, chw, n_p, n_bh = 10, 3, (1,28,28), 7, 2
        lr, epochs, bs = 0.01, 2, 8

        # Code structure
        trainer = self._import_and_test("deep_network", "Trainer", nn.Linear(3, 3), lr, epochs, bs)
        for fn in ["train_all", "train_one_epoch", "predict_torch"]:
            _ = trainer.__getattribute__(fn)

        module = importlib.import_module("src.methods.deep_network")
        for nn_type in ["MLP", "CNN"]:
            model = module.__getattribute__(nn_type)(D if nn_type == "MLP" else 1, C)
            trainer = module.__getattribute__("Trainer")(model, lr, epochs, bs)

            # Functions inputs/outputs
            N = 50
            if nn_type == "MLP":
                train_dataset = TensorDataset(torch.randn(N, D), torch.randint(0, C, (N,)))
                test_dataset = TensorDataset(torch.randn(N, D))
            elif nn_type == "CNN":
                train_dataset = TensorDataset(torch.randn(N, 1, 28, 28), torch.randint(0, C, (N,)))
                test_dataset = TensorDataset(torch.randn(N, 1, 28, 28))

            train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

            # Test Network
            with no_print():
                x, _ = next(iter(train_dataloader))
                preds = model(x)
            self.assertIsInstance(preds, torch.Tensor, f"deep_network.{nn_type}.forward() should output a tensor, not {type(preds)}")
            self.assertEqual(preds.shape, (bs, C), f"deep_network.{nn_type}.forward() output has wrong shape ({preds.shape} != {(bs, C)})")
        # Test Trainer
        with no_print():
            trainer.train_all(train_dataloader)
            pred_labels_test_torch = trainer.predict_torch(test_dataloader)
        self.assertIsInstance(pred_labels_test_torch, torch.Tensor, f"deep_network.Trainer.predict_torch() should output a tensor, not {type(pred_labels_test_torch)}")
        self.assertEqual(pred_labels_test_torch.shape, (N,), f"deep_network.Trainer.predict_torch() output has wrong shape ({pred_labels_test_torch.shape} != {(N,)})")


def warn(msg):
    print(f"\33[33m/!\\ Warning: {msg}\33[39m")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--no-hide', action='store_true', help='Enable printing from the student code')
    args = parser.parse_args()

    project_path = Path(".")

    dir_name = project_path.absolute().name
    if re.match(r'^((\d{6})_){3}project$', dir_name) is None:
        warn("Project folder name must be in the form 'XXXXXX_XXXXXX_XXXXXX_project'")

    if args.no_hide:
        no_print = NoHidePrints
    else:
        no_print = HidePrints

    unittest.main(argv=[''], verbosity=0)
