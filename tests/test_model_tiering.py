import os
import unittest
from unittest.mock import patch

from auto_deal_hunter.infra import config


class ModelTieringTests(unittest.TestCase):
    def test_blank_agent_model_inherits_global_default(self):
        original = config.LLM_MODEL
        try:
            config.LLM_MODEL = "base-model"
            with patch.dict(os.environ, {"SCANNER_MODEL": ""}, clear=False):
                self.assertEqual(config._model_override("SCANNER_MODEL"), "base-model")
        finally:
            config.LLM_MODEL = original

    def test_agent_model_override_wins(self):
        original = config.LLM_MODEL
        try:
            config.LLM_MODEL = "base-model"
            with patch.dict(os.environ, {"PRICER_MODEL": "stronger-model"}, clear=False):
                self.assertEqual(config._model_override("PRICER_MODEL"), "stronger-model")
        finally:
            config.LLM_MODEL = original


if __name__ == "__main__":
    unittest.main()
