import re
from pathlib import Path


def test_all_marked_placement_documentation_examples_execute() -> None:
    root = Path(__file__).parents[2]
    for name in ("placement_recipe.md", "placement_mathematics.md"):
        path = root / "docs" / name
        blocks = re.findall(
            r"<!-- test:[^>]+-->\s*```python\n(.*?)\n```",
            path.read_text(),
            re.DOTALL,
        )
        assert blocks
        namespace = {}
        for block in blocks:
            exec(compile(block, str(path), "exec"), namespace)
