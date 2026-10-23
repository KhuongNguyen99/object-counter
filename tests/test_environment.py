import pytest

from app.environment import load_dotenv


def test_dotenv_preserves_existing_values_and_treats_content_as_data(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# private settings\nKEY=new\nexport EMPTY=\nQUOTED='literal # value' # comment\nRAW=$(do-not-run) # comment\nWIN=\"C:\\models\\countgd\"\n", encoding="utf-8")
    settings = {"KEY": "existing"}
    assert load_dotenv(path, settings)
    assert settings == {"KEY": "existing", "EMPTY": "", "QUOTED": "literal # value",
                        "RAW": "$(do-not-run)", "WIN": r"C:\models\countgd"}


@pytest.mark.parametrize("content", ["BAD KEY=private-value", "KEY='private-value", "KEY='private-value' trailing"])
def test_dotenv_invalid_lines_do_not_echo_values(tmp_path, content):
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_dotenv(path, {})
    assert "private-value" not in str(error.value)


def test_dotenv_missing_is_optional(tmp_path):
    assert not load_dotenv(tmp_path / ".env", {})
