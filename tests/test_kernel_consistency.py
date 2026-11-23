"""Static consistency checks between the CUDA sources and their Python callers.

These run on CPU with no GPU, compiled .so, torch or model weights: they only parse source files.
They guard against two bugs found after the MLA + specdec merge:

* The fused LayerNorm kernel's hard-coded eps (LN_EPSILON) changed from 1e-8 to 1e-5, but the
  specdec correctness gate still built its reference with 1e-8.
* minitorch/cuda_kernel_ops.py called launch_layernorm_bw with one stream, while the C launcher
  takes two, so the second was read from leftover stack memory. ctypes cannot detect this: it
  only checks the call against `argtypes`, never against the C signature.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CUDA_KERNEL_OPS = ROOT / "minitorch" / "cuda_kernel_ops.py"


# ---------------------------------------------------------------------------
# LayerNorm eps
# ---------------------------------------------------------------------------

def _kernel_ln_epsilon() -> float:
    text = (SRC / "layernorm_kernel.cu").read_text()
    m = re.search(r"const\s+float\s+LN_EPSILON\s*=\s*([0-9.eE+-]+)f?\s*;", text)
    assert m, "LN_EPSILON not found in src/layernorm_kernel.cu"
    return float(m.group(1))


def _module_constant(path: Path, name: str):
    """Value of a module-level `NAME = <literal>` assignment, without importing the module."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned at module level in {path}")


def _test_decorator_epsilon() -> float:
    """`self.epsilon = <literal>` in test_utils.TestDecorator (the module imports torch)."""
    for node in ast.walk(ast.parse((ROOT / "test_utils.py").read_text())):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "epsilon"):
            return ast.literal_eval(node.value)
    raise AssertionError("self.epsilon not found in test_utils.py")


def test_specdec_ln_eps_matches_fused_kernel():
    # specdec.test_correctness builds its tight numpy reference with this eps.
    assert _module_constant(ROOT / "specdec" / "common.py", "MINITORCH_LN_EPS") == pytest.approx(
        _kernel_ln_epsilon(), rel=1e-6)


def test_kernel_test_baseline_eps_matches_fused_kernel():
    # kernel_tests/test_layernorm_fw.py compares the kernel against a baseline using this eps.
    assert _test_decorator_epsilon() == pytest.approx(_kernel_ln_epsilon(), rel=1e-6)


def test_correctness_gate_does_not_hard_code_an_eps():
    text = (ROOT / "specdec" / "test_correctness.py").read_text()
    assert "MINITORCH_LN_EPS" in text
    assert not re.search(r"eps\s*=\s*[0-9]", text), "use common.MINITORCH_LN_EPS, not a literal eps"


# ---------------------------------------------------------------------------
# ctypes argtypes / call sites vs C signatures
# ---------------------------------------------------------------------------

_C_DEF = re.compile(
    r"(?:extern\s+\"C\"\s+)?\b(?:void|int|float\s*\*)\s+(\w+)\s*\(([^)]*)\)\s*\{", re.S)


def _c_param_counts() -> dict:
    counts = {}
    for path in SRC.glob("*.cu"):
        text = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
        text = re.sub(r"//[^\n]*", "", text)
        for name, params in _C_DEF.findall(text):
            params = params.strip()
            n = 0 if params in ("", "void") else len(params.split(","))
            counts.setdefault(name, set()).add(n)
    return counts


def _list_len(node, known):
    """Length of an argtypes list expression, or None if it cannot be determined statically."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _list_len(node.left, known), _list_len(node.right, known)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        for seq, k in ((node.left, node.right), (node.right, node.left)):
            n = _list_len(seq, known)
            if n is not None and isinstance(k, ast.Constant) and isinstance(k.value, int):
                return n * k.value
        return None
    if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)
            and isinstance(node.value, ast.Attribute) and node.value.attr == "argtypes"):
        base = known.get(_c_name(node.value.value, {}))
        if base is None:
            return None
        bounds = []
        for b in (node.slice.lower, node.slice.upper):
            bounds.append(None if b is None else ast.literal_eval(b))
        return len(range(base)[slice(*bounds)])
    return None


def _c_name(node, aliases):
    """`lib_x.fn` -> 'fn'; a local alias (fn = lib_x.fn) -> 'fn'; otherwise None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id in _LOADED_LIBS:
        return node.attr
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    return None


def _loaded_libs(tree) -> set:
    """Module-level names bound to ctypes.CDLL(...): only these libraries are actually loaded."""
    libs = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "CDLL"):
            libs.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return libs


_TREE = ast.parse(CUDA_KERNEL_OPS.read_text())
_LOADED_LIBS = _loaded_libs(_TREE)


def _python_bindings():
    """Return ({c_name: argtypes length}, [(c_name, n_call_args, lineno)])."""
    argtypes, calls = {}, []
    for fn in ast.walk(_TREE):
        if not isinstance(fn, ast.FunctionDef):
            continue
        aliases = {}
        # statements in source order, so an argtypes list is known before it is sliced
        for node in (n for stmt in fn.body for n in ast.walk(stmt)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    name = _c_name(node.value, {})
                    if name is not None:
                        aliases[target.id] = name
                elif isinstance(target, ast.Attribute) and target.attr == "argtypes":
                    name = _c_name(target.value, aliases)
                    if name is not None:
                        n = _list_len(node.value, argtypes)
                        assert n is not None, f"cannot size argtypes of {name} (line {node.lineno})"
                        argtypes[name] = n
            elif isinstance(node, ast.Call):
                name = _c_name(node.func, aliases)
                if name is not None and not any(isinstance(a, ast.Starred) for a in node.args):
                    calls.append((name, len(node.args), node.lineno))
    return argtypes, calls


def test_bindings_found():
    argtypes, calls = _python_bindings()
    # sanity: the parser sees the launchers it is meant to check
    for name in ("launch_layernorm", "launch_layernorm_bw", "launch_attn_softmax",
                 "launch_attn_softmax_bw", "launch_mla_forward", "launch_mla_backward",
                 "MatrixMultiply", "MatrixMultiplyResidentB", "DeviceUpload",
                 "tensorMap", "tensorZip", "tensorReduce"):
        assert name in argtypes, name
        assert any(c[0] == name for c in calls), name


def test_argtypes_match_c_signatures():
    c_counts = _c_param_counts()
    argtypes, _ = _python_bindings()
    bad = []
    for name, n in sorted(argtypes.items()):
        if name not in c_counts:
            bad.append(f"{name}: no C definition in src/*.cu")
        elif c_counts[name] != {n}:
            bad.append(f"{name}: argtypes has {n} entries, C signature has {sorted(c_counts[name])}")
    assert not bad, "\n".join(bad)


def test_call_sites_pass_every_declared_argument():
    argtypes, calls = _python_bindings()
    bad = [f"line {line}: {name} called with {n} args, argtypes declares {argtypes[name]}"
           for name, n, line in calls if name in argtypes and n != argtypes[name]]
    assert not bad, "\n".join(bad)
