"""CPU-only checks for experimental patches; no checkpoint or inference required."""

from pathlib import Path
import importlib.util
from types import SimpleNamespace
import unittest

import torch


SPEC = importlib.util.spec_from_file_location("run_strategies", Path(__file__).resolve().parents[1] / "scripts/reliability/strategies/run_strategies.py")
strategies = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(strategies)


class PatchTests(unittest.TestCase):
    def test_yoloe_prototypes_and_restoration(self):
        class Predictor:
            def get_vpe(self, value):
                return value

        class Model:
            def set_classes(self, classes, embeddings=None):
                self.classes, self.embeddings = classes, embeddings

        class Engine:
            def count(self, group, fail=False):
                Predictor().get_vpe(group)
                if fail:
                    raise RuntimeError("reference encoding failed")
                model = Model()
                model.set_classes(["object"], embeddings=group)
                return model

        original_get = Predictor.get_vpe
        original_set = Model.set_classes
        original_count = Engine.count
        groups = [torch.tensor([[[1., 0.], [0., 1.]]]), torch.tensor([[[0., 2.]]])]
        for mode, count in (("yoloe-by-image", 2), ("yoloe-per-object", 3)):
            with strategies.yoloe_prototypes(mode, Model, Predictor, Engine):
                predictor, model = Predictor(), Model()
                for group in groups:
                    predictor.get_vpe(group)
                model.set_classes(["object"], embeddings=groups[0].mean(1, keepdim=True))
                self.assertEqual(model.embeddings.shape, (1, count, 2))
                self.assertEqual(len(model.classes), count)
                self.assertTrue(torch.allclose(model.embeddings.norm(dim=-1), torch.ones(1, count)))
                predictor.get_vpe(groups[1])
                model.set_classes(["object"], embeddings=groups[1])
                self.assertEqual(model.embeddings.shape, (1, 1, 2))
                with self.assertRaisesRegex(RuntimeError, "reference encoding failed"):
                    Engine().count(groups[0], fail=True)
                self.assertEqual(Engine().count(groups[1]).embeddings.shape, (1, 1, 2))
            self.assertIs(Predictor.get_vpe, original_get)
            self.assertIs(Model.set_classes, original_set)
            self.assertIs(Engine.count, original_count)

        with self.assertRaisesRegex(RuntimeError, "forced"):
            with strategies.yoloe_prototypes("yoloe-by-image", Model, Predictor, Engine):
                raise RuntimeError("forced")
        self.assertIs(Predictor.get_vpe, original_get)
        self.assertIs(Model.set_classes, original_set)
        self.assertIs(Engine.count, original_count)

    def test_countgd_tile_geometry_and_restoration(self):
        from app.tiling import make_tiles

        class Engine:
            def count(self):
                return [{"processing": {"tile_size": 800}}]

        original_count = Engine.count
        module = SimpleNamespace(make_tiles=make_tiles, CountGDEngine=Engine)
        baseline = module.make_tiles(1200, 800, 800)
        with strategies.countgd_tiles(512, 0.5, module):
            changed = module.make_tiles(1200, 800, 800)
            self.assertGreater(len(changed), len(baseline))
            self.assertLessEqual(max(tile.width for tile in changed), 512)
            self.assertEqual(Engine().count()[0]["processing"], {"tile_size": 512, "overlap": 0.5})
        self.assertIs(module.make_tiles, make_tiles)
        self.assertIs(Engine.count, original_count)
        with self.assertRaisesRegex(RuntimeError, "forced"):
            with strategies.countgd_tiles(512, 0.5, module):
                raise RuntimeError("forced")
        self.assertIs(module.make_tiles, make_tiles)
        self.assertIs(Engine.count, original_count)


if __name__ == "__main__":
    unittest.main()
