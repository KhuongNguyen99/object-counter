"""Source patch upgrades and isolation of reusable pre-fusion model state."""

import ast
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.patch_countgd import PATCH_VERSION, patch_attention, patch_conditioning, patch_source
from scripts.prepare_countgd import SOURCE_COMMIT, prepare_source


# A small executable model with the upstream section boundaries and mutation contract.
MODEL_SOURCE = '''class GroundingDINO:
    training = False
    text_calls = 0
    reference_calls = 0

    def __init__(self):
        self.head_calls = []
        self.bbox_embed = [Head("box", i, self.head_calls) for i in range(2)]
        self.class_embed = [Head("class", i, self.head_calls) for i in range(2)]

    def backbone(self, image):
        return image.tensors, None

    def forward(self, input_images, input_image_pos_exemplars, pos_exemplars,
                input_images_neg_exemplars, neg_exemplars, captions):
        # encoder texts
        self.text_calls += 1
        tokenized = {"input_ids": torch.tensor([[101, 1012]]).to(input_images.device)}
        one_hot_token = tokenized
        text_dict = {"encoded_text": torch.tensor([float(len(captions[0]))]).to(input_images.device)}

        if isinstance(input_images, (list, torch.Tensor)):
            input_images = nested_tensor_from_tensor_list(input_images)
        features, poss = self.backbone(input_images)

        # Positive exemplars
        self.reference_calls += 1
        text_dict["encoded_text"] += input_image_pos_exemplars.tensors.sum() + pos_exemplars[0].sum()

        srcs = []
        text_dict["encoded_text"].add_(features.sum())
        tokenized["input_ids"].add_(7)
        hs = [torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 10 + i for i in range(2)]
        reference = [torch.full((1, 2, 4), value) for value in (.2, .4, .7)]

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # Used to calculate losses
        out["text_mask"] = torch.ones((1, 4), dtype=torch.bool)
        out["aux_outputs"] = outputs_class[:-1]
        out["token"] = one_hot_token
        out["input_ids"] = tokenized["input_ids"]
        return out
'''

LEGACY_ATTENTION_SOURCE = '''try:
    import MultiScaleDeformableAttention as _C
except ImportError:
    _C = None
    warnings.warn("Custom attention ops unavailable; using the PyTorch fallback.")

class Attention:
    def forward(self, value, spatial_shapes, sampling_locations, attention_weights):
        num_value = value.shape[1]
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        if _C is not None and torch.cuda.is_available() and value.is_cuda:
            return _C.forward(value, spatial_shapes, sampling_locations, attention_weights)
        else:
            return multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
'''
ATTENTION_SOURCE = LEGACY_ATTENTION_SOURCE.replace(
    'except ImportError:\n    _C = None\n    warnings.warn("Custom attention ops unavailable; using the PyTorch fallback.")',
    'except:\n    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")',
).replace("if _C is not None and torch.cuda.is_available()", "if torch.cuda.is_available()")


class Head:
    def __init__(self, kind, layer, calls):
        self.kind, self.layer, self.calls = kind, layer, calls

    def __call__(self, value, text=None):
        self.calls.append((self.kind, self.layer))
        output = value * (self.layer + 1)
        return output if text is None else output + text["encoded_text"]


def legacy_conditioning_source():
    """Reproduce the released v1 methods independently of the current patcher."""
    text_start = MODEL_SOURCE.index("        # encoder texts\n")
    target_start = MODEL_SOURCE.index("        if isinstance(input_images, (list, torch.Tensor)):\n")
    reference_start = MODEL_SOURCE.index("        # Positive exemplars\n")
    output_start = MODEL_SOURCE.index("        srcs = []\n")
    conditioning = (MODEL_SOURCE[text_start:target_start] + MODEL_SOURCE[reference_start:output_start]).replace(
        "input_images.device", "input_image_pos_exemplars.device")
    return MODEL_SOURCE + '''
    def prepare_conditioning(self, input_image_pos_exemplars, pos_exemplars,
                             input_images_neg_exemplars, neg_exemplars, captions):
        if self.training:
            raise RuntimeError('Reference conditioning cache requires eval mode.')
''' + conditioning + '''        return text_dict, tokenized

    def forward_conditioned(self, input_images, conditioning):
        if self.training:
            raise RuntimeError('Reference conditioning cache requires eval mode.')
        # Fusion updates text state; every target starts from the same unfused tokens.
        text_dict = {key: value.clone() for key, value in conditioning[0].items()}
        tokenized = copy.deepcopy(conditioning[1])
        one_hot_token = tokenized
''' + MODEL_SOURCE[target_start:reference_start] + MODEL_SOURCE[output_start:]


@torch.inference_mode()
def test_cached_forward_preserves_original_and_starts_each_target_from_unfused_state():
    patched = patch_conditioning(MODEL_SOURCE)
    assert patch_conditioning(patched) == patched
    original_forward = ast.parse(MODEL_SOURCE).body[0].body[-1]
    patched_forward = next(node for node in ast.parse(patched).body[0].body
                           if isinstance(node, ast.FunctionDef) and node.name == "forward")
    assert ast.dump(original_forward) == ast.dump(patched_forward)
    namespace = {"torch": torch, "copy": copy, "Head": Head, "inverse_sigmoid": torch.logit}
    exec(patched, namespace)
    model = namespace["GroundingDINO"]()
    reference = SimpleNamespace(tensors=torch.tensor([2.]), device="cpu")
    boxes = [torch.tensor([[1., 2., 3., 4.]])]
    prepared = model.prepare_conditioning(reference, boxes, [], [], ["ring . "])
    original_state = copy.deepcopy(prepared)
    targets = [SimpleNamespace(tensors=torch.tensor([value]), device="cpu") for value in (3., 8., 3.)]
    cached = [model.forward_conditioned(target, prepared) for target in targets]
    assert (model.text_calls, model.reference_calls) == (1, 1)
    assert model.head_calls == [("class", 1), ("box", 1)] * len(targets)
    for target, result in zip(targets, cached, strict=True):
        uncached = model.forward(target, reference, boxes, [], [], ["ring . "])
        assert set(result) == {"pred_logits", "pred_boxes", "input_ids"}
        for name, value in result.items():
            torch.testing.assert_close(value, uncached[name], rtol=0, atol=0)
    torch.testing.assert_close(prepared[0]["encoded_text"], original_state[0]["encoded_text"], rtol=0, atol=0)
    torch.testing.assert_close(prepared[1]["input_ids"], original_state[1]["input_ids"], rtol=0, atol=0)
    assert cached[0]["pred_logits"] is not cached[2]["pred_logits"]
    model.training = True
    with pytest.raises(RuntimeError, match="eval mode"):
        model.prepare_conditioning(reference, boxes, [], [], ["ring . "])
    with pytest.raises(RuntimeError, match="eval mode"):
        model.forward_conditioned(targets[0], prepared)


def test_conditioning_patch_rejects_changed_boundaries_and_existing_patch():
    with pytest.raises(ValueError, match="did not match"):
        patch_conditioning(MODEL_SOURCE.replace("# Positive exemplars", "# Changed upstream"))
    patched = patch_conditioning(MODEL_SOURCE).replace("value.clone()", "value")
    with pytest.raises(ValueError, match="does not match this version"):
        patch_conditioning(patched)
    with pytest.raises(ValueError, match="output heads did not match"):
        patch_conditioning(MODEL_SOURCE.replace("reference[:-1]", "reference[1:]"))
    with pytest.raises(ValueError, match="does not match this version"):
        patch_conditioning(legacy_conditioning_source().replace("value.clone()", "value"))


def test_v1_cached_methods_upgrade_to_v2_without_changing_official_forward():
    legacy = legacy_conditioning_source()
    patched = patch_conditioning(legacy)
    assert patched.startswith(MODEL_SOURCE)
    assert patched == patch_conditioning(MODEL_SOURCE)
    assert patch_conditioning(patched) == patched


def test_attention_patch_is_idempotent_and_rejects_unknown_source():
    patched = patch_attention(ATTENTION_SOURCE)
    assert patch_attention(patched) == patched
    assert "_C is not None" in patched
    assert patch_attention(LEGACY_ATTENTION_SOURCE) == patched
    with pytest.raises(ValueError, match="compatibility patch"):
        patch_attention(ATTENTION_SOURCE.replace("value.is_cuda", "value.device"))
    with pytest.raises(ValueError, match="compatibility patch"):
        patch_attention(patched.replace("sum(height * width", "sum(height + width"))


def test_attention_fallback_receives_one_shape_list_and_preserves_size_validation():
    # Tensor indexing is forbidden: Python split/reshape sizes must use one transfer.
    class Shapes:
        calls = 0

        def tolist(self):
            self.calls += 1
            return [[2, 3], [1, 2]]

    captured = []

    def fallback(value, shapes, locations, weights):
        captured.append(shapes)
        return value.split([height * width for height, width in shapes], dim=1)

    import warnings
    namespace = {"torch": torch, "warnings": warnings, "multi_scale_deformable_attn_pytorch": fallback}
    with pytest.warns(UserWarning, match="PyTorch fallback"):
        exec(patch_attention(ATTENTION_SOURCE), namespace)
    attention = namespace["Attention"]()
    shapes = Shapes()
    value = torch.arange(8).reshape(1, 8, 1)
    result = attention.forward(value, shapes, None, None)
    assert shapes.calls == 1 and captured == [[[2, 3], [1, 2]]]
    torch.testing.assert_close(torch.cat(result, dim=1), value, rtol=0, atol=0)
    with pytest.raises(AssertionError):
        attention.forward(value[:, :7], shapes, None, None)
    assert shapes.calls == 2 and len(captured) == 1


@pytest.mark.parametrize("source_version", ["fresh", "v1", "v2"])
def test_existing_commit_stamped_source_upgrades_without_download(tmp_path, monkeypatch, source_version):
    source = tmp_path / "countgd"
    model_dir = source / "models/GroundingDINO"
    model_dir.mkdir(parents=True)
    attention, model = model_dir / "ms_deform_attn.py", model_dir / "groundingdino_app.py"
    attention.write_text({"fresh": ATTENTION_SOURCE, "v1": LEGACY_ATTENTION_SOURCE,
                          "v2": patch_attention(ATTENTION_SOURCE)}[source_version])
    model.write_text({"fresh": MODEL_SOURCE, "v1": legacy_conditioning_source(),
                      "v2": patch_conditioning(MODEL_SOURCE)}[source_version])
    if source_version != "fresh":
        (source / ".counter-source-patches").write_text(
            "reference-conditioning-v1\n" if source_version == "v1" else PATCH_VERSION + "\n")
    (source / ".counter-source-commit").write_text(SOURCE_COMMIT + "\n")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("Unexpected download"))
    prepare_source(source)
    after = (attention.read_text(), model.read_text())
    assert "def forward_conditioned" in after[1]
    assert (source / ".counter-source-patches").read_text().strip() == PATCH_VERSION
    prepare_source(source)
    assert (attention.read_text(), model.read_text()) == after


def test_source_drift_is_rejected_before_any_file_is_updated(tmp_path):
    model_dir = tmp_path / "models/GroundingDINO"
    model_dir.mkdir(parents=True)
    attention = model_dir / "ms_deform_attn.py"
    attention.write_text(ATTENTION_SOURCE)
    (model_dir / "groundingdino_app.py").write_text(MODEL_SOURCE.replace("# encoder texts", "# Changed upstream"))
    with pytest.raises(ValueError, match="did not match"):
        patch_source(tmp_path)
    assert attention.read_text() == ATTENTION_SOURCE
    assert not (tmp_path / ".counter-source-patches").exists()
