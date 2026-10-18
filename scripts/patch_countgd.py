"""Small, checked inference patches for the pinned official CountGD++ source."""

import ast
from pathlib import Path

PATCH_VERSION = "gpu-execution-v2"

_ORIGINAL_HEADS = '''        # deformable-detr-like anchor update
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

'''

_INFERENCE_HEADS = '''        # Only final decoder predictions are consumed during inference.
        return {
            "pred_logits": self.class_embed[-1](hs[-1], text_dict),
            "pred_boxes": (self.bbox_embed[-1](hs[-1]) + inverse_sigmoid(reference[-2])).sigmoid(),
            "input_ids": tokenized["input_ids"],
        }
'''


def patch_attention(content: str) -> str:
    replacements = (
        ('except:\n    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")',
         'except ImportError:\n    _C = None\n    warnings.warn("Custom attention ops unavailable; using the PyTorch fallback.")'),
        ("if torch.cuda.is_available() and value.is_cuda:",
         "if _C is not None and torch.cuda.is_available() and value.is_cuda:"),
        ("        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value",
         "        # Keep tensor dimensions on the host instead of synchronizing each scalar.\n"
         "        spatial_shapes_list = spatial_shapes.tolist()\n"
         "        assert sum(height * width for height, width in spatial_shapes_list) == num_value"),
        ("                value, spatial_shapes, sampling_locations, attention_weights\n",
         "                value, spatial_shapes_list, sampling_locations, attention_weights\n"),
    )
    for old, new in replacements:
        if content.count(old) == 1 and new not in content:
            content = content.replace(old, new)
        elif old in content or content.count(new) != 1:
            raise ValueError("Pinned attention source did not match the reviewed compatibility patch")
    return content


def _between(content: str, start: str, end: str) -> str:
    if content.count(start) != 1 or content.count(end) != 1:
        raise ValueError("Pinned CountGD++ forward did not match the conditioning patch")
    start_index, end_index = content.index(start), content.index(end)
    if start_index >= end_index:
        raise ValueError("Pinned CountGD++ forward sections are out of order")
    return content[start_index:end_index]


def patch_conditioning(content: str) -> str:
    """Keep official forward intact; derive cached inference from its exact blocks."""
    parsed = ast.parse(content)
    model = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "GroundingDINO")
    methods = {node.name: node for node in model.body if isinstance(node, ast.FunctionDef)}
    lines = content.splitlines(keepends=True)

    def source(node):
        return "".join(lines[node.lineno - 1:node.end_lineno])

    forward = source(methods["forward"])
    target_start = "        if isinstance(input_images, (list, torch.Tensor)):\n"
    reference_start = "        # Positive exemplars\n"
    output_start = "        srcs = []\n"
    text = _between(forward, "        # encoder texts\n", target_start)
    target = _between(forward, target_start, reference_start)
    reference = _between(forward, reference_start, output_start)
    output = forward[forward.index(output_start):].rstrip() + "\n"
    heads_start = "        # deformable-detr-like anchor update\n"
    if _between(output, heads_start, "        # Used to calculate losses\n") != _ORIGINAL_HEADS:
        raise ValueError("Pinned CountGD++ output heads did not match the inference patch")
    inference_output = _between(output, output_start, heads_start) + _INFERENCE_HEADS
    conditioning = (text + reference).replace("input_images.device", "input_image_pos_exemplars.device")
    prepared = (
        "    def prepare_conditioning(self, input_image_pos_exemplars, pos_exemplars,\n"
        "                             input_images_neg_exemplars, neg_exemplars, captions):\n"
        "        if self.training:\n"
        "            raise RuntimeError('Reference conditioning cache requires eval mode.')\n"
        + conditioning
        + "        return text_dict, tokenized\n"
    )
    conditioned_prefix = (
        "    def forward_conditioned(self, input_images, conditioning):\n"
        "        if self.training:\n"
        "            raise RuntimeError('Reference conditioning cache requires eval mode.')\n"
        "        # Fusion updates text state; every target starts from the same unfused tokens.\n"
        "        text_dict = {key: value.clone() for key, value in conditioning[0].items()}\n"
        "        tokenized = copy.deepcopy(conditioning[1])\n"
    )
    legacy_conditioned = conditioned_prefix + "        one_hot_token = tokenized\n" + target + output
    conditioned = conditioned_prefix + target + inference_output
    expected = {"prepare_conditioning": prepared, "forward_conditioned": conditioned}
    if any(name in methods for name in expected):
        if (not all(name in methods for name in expected)
                or source(methods["prepare_conditioning"]).rstrip() != prepared.rstrip()
                or source(methods["forward_conditioned"]).rstrip() not in
                (conditioned.rstrip(), legacy_conditioned.rstrip())):
            raise ValueError("Existing CountGD++ conditioning patch does not match this version")
        if source(methods["forward_conditioned"]).rstrip() == conditioned.rstrip():
            return content
        method = methods["forward_conditioned"]
        patched = "".join(lines[:method.lineno - 1]) + conditioned + "".join(lines[method.end_lineno:])
    else:
        end = methods["forward"].end_lineno
        patched = "".join(lines[:end]) + "\n" + prepared + "\n" + conditioned + "".join(lines[end:])
    ast.parse(patched)
    return patched


def patch_source(source: Path) -> None:
    attention = source / "models/GroundingDINO/ms_deform_attn.py"
    model = source / "models/GroundingDINO/groundingdino_app.py"
    originals = (attention.read_text(), model.read_text())
    patched = (patch_attention(originals[0]), patch_conditioning(originals[1]))
    for path, before, after in zip((attention, model), originals, patched, strict=True):
        if before != after:
            path.write_text(after)
    (source / ".counter-source-patches").write_text(PATCH_VERSION + "\n")
