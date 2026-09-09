import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    PROJECT_ROOT / "作者风格版" / "1-OASTER真实ERP.py",
    PROJECT_ROOT / "作者风格版" / "2-OASTER仿真.py",
)


def test_author_style_scripts_are_linear_cell_scripts():
    forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script))

        assert source.count("#%%") + source.count("# %%") >= 10
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree))
