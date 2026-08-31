import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "unsloth"
    / "models"
    / "_flash_attention_compat.py"
)


def _load_compat_module():
    spec = importlib.util.spec_from_file_location(
        "unsloth_flash_attention_compat_test_target",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FlashAttentionCompatTests(unittest.TestCase):
    def test_non_2d_position_ids_do_not_select_packed_sequence_path(self):
        compat = _load_compat_module()
        calls = []

        def original(position_ids, batch_size):
            calls.append((position_ids, batch_size))
            return True

        transformers = ModuleType("transformers")
        transformers.__path__ = []
        fa_utils = ModuleType("transformers.modeling_flash_attention_utils")
        fa_utils._is_packed_sequence = original
        transformers.modeling_flash_attention_utils = fa_utils

        consumer = ModuleType("transformers.some_consumer")
        consumer._is_packed_sequence = original

        modules = {
            "transformers": transformers,
            "transformers.modeling_flash_attention_utils": fa_utils,
            "transformers.some_consumer": consumer,
        }

        with patch.dict(sys.modules, modules, clear=False):
            self.assertTrue(
                compat.fix_transformers_flash_attention_packed_sequence_detection()
            )
            patched = fa_utils._is_packed_sequence

            # Already-imported references are updated as well.
            self.assertIs(consumer._is_packed_sequence, patched)

            qwen35_position_ids = SimpleNamespace(ndim=3)
            self.assertFalse(patched(qwen35_position_ids, 1))
            self.assertEqual(calls, [])

            ordinary_position_ids = SimpleNamespace(ndim=2)
            self.assertTrue(patched(ordinary_position_ids, 1))
            self.assertEqual(calls, [(ordinary_position_ids, 1)])

            # The compatibility fix is safe to call repeatedly.
            self.assertTrue(
                compat.fix_transformers_flash_attention_packed_sequence_detection()
            )
            self.assertIs(fa_utils._is_packed_sequence, patched)

    def test_tensor_like_dim_method_is_supported(self):
        compat = _load_compat_module()
        original_called = False

        def original(position_ids, batch_size):
            nonlocal original_called
            original_called = True
            return True

        patched = compat._guard_non_2d_position_ids(original)

        class PositionIds:
            def dim(self):
                return 3

        self.assertFalse(patched(PositionIds(), 1))
        self.assertFalse(original_called)


if __name__ == "__main__":
    unittest.main()
