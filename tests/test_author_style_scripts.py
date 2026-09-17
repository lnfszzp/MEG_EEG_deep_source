import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    PROJECT_ROOT / "作者风格版" / "1-OASTER真实ERP.py",
    PROJECT_ROOT / "作者风格版" / "2-OASTER仿真.py",
    PROJECT_ROOT / "作者风格版" / "3-ERP四类任务验证.py",
    PROJECT_ROOT / "作者风格版" / "4-DBS频谱定位.py",
)


def test_author_style_scripts_are_linear_cell_scripts():
    forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script))

        assert source.count("#%%") + source.count("# %%") >= 10
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree))


def test_four_paradigm_reuses_validated_ds006035_preprocessing():
    source = SCRIPTS[2].read_text(encoding="utf-8")
    events = source.index("ds_events = np.zeros")
    interpolate = source.index("real_pipeline.interpolate_stimulation_artifacts")
    filtering = source.index("ds_raw.filter")

    assert events < interpolate < filtering
    assert "tmin=-0.002" in source[interpolate:filtering]
    assert "tmax=0.008" in source[interpolate:filtering]
    assert "protected.whitening_matrix(\n        ds_eeg_noise_check" in source
    assert "protected.whitening_matrix(\n        ds_mag_noise_check" in source
