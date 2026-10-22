"""Local counting models and pinned checkpoint integrity checks."""

import hashlib
import os
from pathlib import Path
import zipfile


ASSET_RELEASE = "v8.4.0"
# Sizes and SHA-256 digests published by the official GitHub release API.
MODEL_ASSETS = {
    "yoloe-26n-seg.pt": (11710443, "1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b"),
    "yoloe-26s-seg.pt": (31072171, "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654"),
    "yoloe-26m-seg.pt": (70443185, "585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83"),
    "yoloe-26l-seg.pt": (79417413, "a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9"),
    "yoloe-26x-seg.pt": (171640453, "d08d390a08f98195f7c87807839fe4ff93a5491645fef1bc3bf0700efafdd639"),
}
MODEL_FILENAMES = tuple(MODEL_ASSETS)
FASTSAM_MODEL_ID = "fastsam-s"
FASTSAM_FILENAME = "FastSAM-s.pt"
# Official Ultralytics assets release v8.4.0, used only for reference proposals.
FASTSAM_ASSETS = {
    FASTSAM_FILENAME: (23851578, "c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844"),
}
SIZE_NAMES = {"n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "Extra large"}
COUNTGD_MODEL_ID = "countgd-plusplus"
CROSSCHECK_MODEL_ID = "crosscheck-countgd-yoloe26x"
PETDINO_MODEL_ID = "pet-dino-swin-t"
SAM3COUNT_MODEL_ID = "sam3count"
SAM3_MODEL_ID = "sam3"
# Approved official Meta base and the SAM3Count authors' public dense-counting overlay.
SAM3COUNT_ASSETS = {
    "sam3.pt": (3450062241, "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"),
    "sam3_ft.pt": (3606641522, "9c2383bab0584e1f0600b45da294434f573644566c7c615b2998338286b98615"),
}
# Official fuweifu/PET-DINO checkpoint at a8d8cddbe752bbcc3df299b2fe56d0aa67a26776.
PETDINO_ASSETS = {
    "pet_dino_swin-t_8xb4_12e_obj365.pth": (
        2309043797, "beff20e86a46008b637edf6f519f100cb69981d1c116a03959663d276adc1736"),
}
# Downloaded from the official CountGD++ checkpoint and its official BERT preparation recipe.
COUNTGD_ASSETS = {
    "countgd_plusplus.pth": (1250122586, "b89395d79711c9b9f13f3ef80cd2e528386fb1d70453f99443be4eccab33592f"),
    "bert-base-uncased/config.json": (658, "3f4cbb600213e1a05e68a0bb53ee88b61d5f9d0471165d909d5a395aae2d3b76"),
    "bert-base-uncased/model.safetensors": (435588776, "5875f83030335d194f35b15a32e7f4e654aa302aa83af032a3f36d035dcaf8af"),
    "bert-base-uncased/special_tokens_map.json": (125, "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
    "bert-base-uncased/vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    "bert-base-uncased/tokenizer.json": (711396, "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    "bert-base-uncased/tokenizer_config.json": (1190, "f62a57a75856b93282501c92a86f62b169997c81e93cf6f75b7cc15d6285968e"),
}


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "models")).resolve()


def countgd_dir() -> Path:
    return model_dir() / "countgd"


def petdino_dir() -> Path:
    return model_dir() / "petdino"


def sam3count_dir() -> Path:
    return model_dir() / "sam3count"


def validate_sam3count_assets(directory: Path | None = None) -> None:
    directory = directory or sam3count_dir()
    for name, (size, digest) in SAM3COUNT_ASSETS.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"SAM3Count asset {name} is missing. Prepare the approved checkpoints first.")
        if path.stat().st_size != size:
            raise ValueError(f"SAM3Count asset {name} is incomplete or invalid.")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"SAM3Count asset {name} failed SHA-256 validation.")


def validate_sam3_base_asset(directory: Path | None = None) -> None:
    directory = directory or sam3count_dir()
    name = "sam3.pt"
    size, digest = SAM3COUNT_ASSETS[name]
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError("SAM3 base checkpoint is missing.")
    if path.stat().st_size != size:
        raise ValueError("SAM3 base checkpoint is incomplete or invalid.")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("SAM3 base checkpoint failed SHA-256 validation.")


def validate_petdino_assets(directory: Path | None = None) -> None:
    directory = directory or petdino_dir()
    assets = [(directory / name, size, digest) for name, (size, digest) in PETDINO_ASSETS.items()]
    assets.extend((countgd_dir() / name, size, digest) for name, (size, digest) in COUNTGD_ASSETS.items()
                  if name.startswith("bert-base-uncased/"))
    for path, size, digest in assets:
        if not path.is_file():
            raise FileNotFoundError(f"PET-DINO asset {path.name} is missing. Run python scripts/prepare_petdino.py.")
        if path.stat().st_size != size:
            raise ValueError(f"PET-DINO asset {path.name} is incomplete or invalid.")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"PET-DINO asset {path.name} failed SHA-256 validation.")


def validate_countgd_assets(directory: Path | None = None) -> None:
    directory = directory or countgd_dir()
    for filename, (expected_bytes, expected_hash) in COUNTGD_ASSETS.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"CountGD++ asset {filename} is missing. Run python scripts/prepare_countgd.py.")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"CountGD++ asset {filename} is incomplete or invalid.")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                raise ValueError(f"CountGD++ asset {filename} failed SHA-256 validation.")


def checkpoint_path(model_id: str) -> Path:
    filename = f"{model_id}.pt"
    if filename not in MODEL_ASSETS:
        raise ValueError(f"Unknown model: {model_id}")
    return model_dir() / filename


def validate_checkpoint(path: Path, filename: str | None = None) -> None:
    """Validate trusted release bytes before torch loads its pickle checkpoint."""
    filename = filename or path.name
    expected_bytes, expected_hash = (MODEL_ASSETS | FASTSAM_ASSETS)[filename]
    if not path.is_file():
        raise FileNotFoundError(f"Model {filename} is missing. Run python scripts/download_models.py first.")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Model {filename} is incomplete or invalid. Run python scripts/download_models.py to repair it.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_hash or not zipfile.is_zipfile(path):
        raise ValueError(f"Model {filename} failed SHA-256 validation. Run python scripts/download_models.py to repair it.")


def list_models() -> list[dict]:
    paths = [countgd_dir() / filename for filename in COUNTGD_ASSETS]
    source_ready = (Path(os.environ.get("COUNTGD_SOURCE_DIR", "/opt/countgd")) / "cfg_app.py").is_file()
    return [{
        "id": COUNTGD_MODEL_ID, "name": "CountGD++",
        "family": "CountGD++", "size": "base", "default_confidence": 0.80,
        "reference_mode": "per_image", "required_reference_boxes": 10,
        "available": source_ready and all(path.is_file() and path.stat().st_size == expected_bytes
                         for path, (expected_bytes, _) in zip(paths, COUNTGD_ASSETS.values())),
        "bytes": sum(path.stat().st_size for path in paths if path.is_file()),
    }]
