"""CI gate: no agent-reachable tool may route untrusted free-text through
`just` recipe `{{param}}` interpolation (Finding 1/7, CWE-78).

Parses every tool module's AST and flags any `just_run(...)` positional arg
(after the recipe name) that is NOT provably safe. Safe forms:
  - string/number literal, or ternary whose branches are literals;
  - `str(...)`, `int(...)`, `float(...)` calls (numerics) and `*args` expansion;
  - a name produced by a sanitizer (validate_identifier/resolve_*/validate_url);
  - a name guarded by an enum-membership check earlier in the function, i.e.
    `if NAME not in {...}/SET: return ...` (the value is one of a fixed set);
  - `tmp.name` (a server-generated NamedTemporaryFile path).

Untrusted free-text must be passed via `extra_env=` (env vars), which `just`
does not interpolate. This encodes the contract so a regression fails CI.
"""
import ast
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "strands_cosmos" / "tools"

_SANITIZERS = {"validate_identifier", "resolve_in_workspace", "resolve_output_path",
               "validate_url", "validate_nats_subject"}


def _sanitized_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        # assigned from a sanitizer call
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            cn = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
            if cn in _SANITIZERS:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
        # enum-membership guard: `if NAME not in <set/name>: <body that returns>`
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            t = node.test
            if (len(t.ops) == 1 and isinstance(t.ops[0], ast.NotIn)
                    and isinstance(t.left, ast.Name)):
                # body must short-circuit (return) for the guard to constrain
                if any(isinstance(b, ast.Return) for b in node.body):
                    names.add(t.left.id)
        # regex fullmatch guard: `if not re.fullmatch(...NAME...): return`
        if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) \
                and isinstance(node.test.op, ast.Not) \
                and isinstance(node.test.operand, ast.Call):
            call = node.test.operand
            fn = call.func
            fname = getattr(fn, "attr", "") or (fn.id if isinstance(fn, ast.Name) else "")
            if "fullmatch" in fname or "match" in fname:
                for a in call.args:
                    # str(NAME) or NAME
                    if isinstance(a, ast.Name):
                        names.add(a.id)
                    elif isinstance(a, ast.Call) and a.args and isinstance(a.args[0], ast.Name):
                        names.add(a.args[0].id)
    return names


def _is_literal_ternary(arg: ast.expr) -> bool:
    return (isinstance(arg, ast.IfExp)
            and isinstance(arg.body, ast.Constant)
            and isinstance(arg.orelse, ast.Constant))


def _arg_is_safe(arg: ast.expr, sanitized: set[str]) -> bool:
    if isinstance(arg, ast.Constant):
        return True
    if isinstance(arg, ast.Starred):
        return True
    if _is_literal_ternary(arg):
        return True
    if isinstance(arg, ast.Name):
        return arg.id in sanitized
    if isinstance(arg, ast.Attribute):
        # tmp.name  (NamedTemporaryFile generated path)
        return arg.attr == "name"
    if isinstance(arg, ast.Call):
        fn = arg.func
        fname = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
        if fname in {"str", "int", "float"}:
            return True
    return False


def test_no_untrusted_just_interpolation():
    offenders = []
    for py in sorted(TOOLS_DIR.glob("*.py")):
        if py.name in ("_common.py", "_security.py", "__init__.py"):
            continue
        tree = ast.parse(py.read_text())
        for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            sanitized = _sanitized_names(func)
            for node in ast.walk(func):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "just_run"):
                    for i, a in enumerate(node.args[1:], start=1):
                        if not _arg_is_safe(a, sanitized):
                            offenders.append(
                                f"{py.name}:{node.lineno} just_run positional #{i}: "
                                f"{ast.dump(a)[:80]}"
                            )
    assert not offenders, (
        "Untrusted args routed through `just` interpolation:\n  "
        + "\n  ".join(offenders)
    )
