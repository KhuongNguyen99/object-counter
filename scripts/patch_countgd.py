"""Small, checked inference patches for the pinned official CountGD++ source."""

import ast
from pathlib import Path

PATCH_VERSION = "reference-conditioning-v1"


def patch_attention(content: str) -> str:
    replacements = (
        ('except:\n    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")',
         'except ImportError:\n    _C = None\n    warnings.warn("Custom attention ops unavailable; using the PyTorch fallback.")'),
        ("if torch.cuda.is_available() and value.is_cuda:",
         "if _C is not None and torch.cuda.is_available() and value.is_cuda:"),
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
    conditioning = (text + reference).replace("input_images.device", "input_image_pos_exemplars.device")
    prepared = (
        "    def prepare_conditioning(self, input_image_pos_exemplars, pos_exemplars,\n"
        "                             input_images_neg_exemplars, neg_exemplars, captions):\n"
        "        if self.training:\n"
        "            raise RuntimeError('Reference conditioning cache requires eval mode.')\n"
        + conditioning
        + "        return text_dict, tokenized\n"
    )
    conditioned = (
        "    def forward_conditioned(self, input_images, conditioning):\n"
        "        if self.training:\n"
        "            raise RuntimeError('Reference conditioning cache requires eval mode.')\n"
        "        # Fusion updates text state; every target starts from the same unfused tokens.\n"
        "        text_dict = {key: value.clone() for key, value in conditioning[0].items()}\n"
        "        tokenized = copy.deepcopy(conditioning[1])\n"
        "        one_hot_token = tokenized\n"
        + target + output
    )
    expected = {"prepare_conditioning": prepared, "forward_conditioned": conditioned}
    if any(name in methods for name in expected):
        if not all(name in methods and source(methods[name]).rstrip() == value.rstrip()
                   for name, value in expected.items()):
            raise ValueError("Existing CountGD++ conditioning patch does not match this version")
        return content
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
