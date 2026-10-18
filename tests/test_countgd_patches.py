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
        return {"value": text_dict["encoded_text"], "input_ids": tokenized["input_ids"], "token": one_hot_token}
'''

ATTENTION_SOURCE = '''try:
    import MultiScaleDeformableAttention as _C
except:
    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")

if torch.cuda.is_available() and value.is_cuda:
    pass
'''


@torch.inference_mode()
def test_cached_forward_preserves_original_and_starts_each_target_from_unfused_state():
    patched = patch_conditioning(MODEL_SOURCE)
    assert patch_conditioning(patched) == patched
    original_forward = ast.parse(MODEL_SOURCE).body[0].body[-1]
    patched_forward = next(node for node in ast.parse(patched).body[0].body
                           if isinstance(node, ast.FunctionDef) and node.name == "forward")
    assert ast.dump(original_forward) == ast.dump(patched_forward)
    namespace = {"torch": torch, "copy": copy}
    exec(patched, namespace)
    model = namespace["GroundingDINO"]()
    reference = SimpleNamespace(tensors=torch.tensor([2.]), device="cpu")
    boxes = [torch.tensor([[1., 2., 3., 4.]])]
    prepared = model.prepare_conditioning(reference, boxes, [], [], ["ring . "])
    original_state = copy.deepcopy(prepared)
    targets = [SimpleNamespace(tensors=torch.tensor([value]), device="cpu") for value in (3., 8., 3.)]
    cached = [model.forward_conditioned(target, prepared) for target in targets]
    assert (model.text_calls, model.reference_calls) == (1, 1)
    for target, result in zip(targets, cached, strict=True):
        uncached = model.forward(target, reference, boxes, [], [], ["ring . "])
        torch.testing.assert_close(result["value"], uncached["value"], rtol=0, atol=0)
        torch.testing.assert_close(result["input_ids"], uncached["input_ids"], rtol=0, atol=0)
        assert result["token"]["input_ids"] is result["input_ids"]
    torch.testing.assert_close(prepared[0]["encoded_text"], original_state[0]["encoded_text"], rtol=0, atol=0)
    torch.testing.assert_close(prepared[1]["input_ids"], original_state[1]["input_ids"], rtol=0, atol=0)
    assert cached[0]["value"] is not cached[2]["value"]
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


def test_attention_patch_is_idempotent_and_rejects_unknown_source():
    patched = patch_attention(ATTENTION_SOURCE)
    assert patch_attention(patched) == patched
    assert "_C is not None" in patched
    with pytest.raises(ValueError, match="compatibility patch"):
        patch_attention(ATTENTION_SOURCE.replace("value.is_cuda", "value.device"))


def test_existing_commit_stamped_source_upgrades_without_download(tmp_path, monkeypatch):
    source = tmp_path / "countgd"
    model_dir = source / "models/GroundingDINO"
    model_dir.mkdir(parents=True)
    attention, model = model_dir / "ms_deform_attn.py", model_dir / "groundingdino_app.py"
    attention.write_text(patch_attention(ATTENTION_SOURCE))
    model.write_text(MODEL_SOURCE)
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
