"""Validate inherited bytes and reconstruct final summaries from public rows."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    inventory = json.loads((root / "provenance/inherited-public-files.json").read_text())
    for row in inventory:
        data = (root / row["path"]).read_bytes()
        assert len(data) == row["bytes"], row["path"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 22
    for path in notebooks:
        for i, cell in enumerate(json.loads(path.read_text())["cells"]):
            if cell["cell_type"] == "code":
                assert not cell.get("outputs")
                code = "\n".join(line for line in "".join(cell["source"]).splitlines()
                                 if not line.lstrip().startswith(("!", "%")))
                compile(code, path.name + f":cell{i}", "exec")
    report = json.loads((root / "analysis/private-artifact-audit.json").read_text())
    assert report["source"]["present_files_verified"] == 45
    frame = pd.read_csv(root / "results/hyperkt_stage10/per_seed_test_metrics.csv")
    original = pd.read_csv(root / "results/hyperkt_stage10/five_seed_test_summary.csv", header=[0, 1], index_col=0)
    original = original.loc[original.index.notna()]
    assert len(frame) == 15
    assert set(frame.seed) == {42, 123, 2024, 3407, 7777}
    for variant, group in frame.groupby("variant"):
        assert len(group) == 5
        for metric in ("auc", "nll", "brier", "f1", "ece15", "accuracy", "precision", "recall"):
            assert np.isclose(group[metric].mean(), float(original.loc[variant, (metric, "mean")]), atol=1e-12, rtol=0)
            assert np.isclose(group[metric].std(ddof=1), float(original.loc[variant, (metric, "std")]), atol=1e-12, rtol=0)
    notebook = json.loads((root / "notebooks/HyperKT_Stage10_Frozen_Heldout_Test_Evaluation_Colab.ipynb").read_text())
    code = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith(("!", "%")))
    names = {"CausalBlock", "HyperKT", "DKT", "HyperKTGated"}
    expected = [ast.dump(n, include_attributes=False) for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name in names]
    actual = [ast.dump(n, include_attributes=False) for n in ast.parse((root / "src/hyperkt/models.py").read_text()).body if isinstance(n, ast.ClassDef)]
    assert actual == expected
    for path in (root / "results").rglob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            assert not ({"user_id", "y", "p", "history_skill", "history_correct", "target_correct"} & set(archive.files)), path
    return {"inherited_files": len(inventory), "notebooks": len(notebooks), "final_metric_rows": len(frame),
            "source_manifest_matches": 45, "exported_model_classes_AST_identical": True,
            "note": "Public structural/table validation, not an independent rerun of private predictions or bootstrap intervals."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
