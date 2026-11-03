#!/usr/bin/env python3
"""Run CPU unit tests and the retained mathematical regression checks."""
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
torch.set_num_threads(2)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    from tests import test_vcr, test_ns_cdn
    extra = 0
    for module in (test_vcr, test_ns_cdn):
        for name in sorted(vars(module)):
            function = getattr(module, name)
            if name.startswith('test_') and callable(function):
                function()
                extra += 1
                print('PASS:', module.__name__ + '.' + name)
    print('Passed %d unit tests and %d module checks; skipped %d.' % (
        result.testsRun - len(result.skipped), extra, len(result.skipped)))
