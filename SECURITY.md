# Security

## Reporting

Report vulnerabilities privately to the maintainers (do not open a public
issue). Include a minimal reproduction and the affected version.

## Threat model

`strands-cosmos` tools are **agent-reachable**: their arguments originate from
LLM output and are therefore **untrusted** (prompt injection, poisoned upstream
content, model misalignment). The hardening model below treats every tool
argument as adversarial.

### Structural fix for the `just` interpolation RCE (CWE-78)

`just` `{{param}}` is *text templating*, not parameter passing — a quote,
triple-quote, or `;` in a value broke out of a recipe and executed
attacker-controlled shell/Python. The fix:

1. **Hot/data-plane tools call the binary directly** via an argv list
   (`_security.safe_run`, `shell=False`) — never through `just`. This applies to
   `video_probe`, `video_extract_frames`, `nats_publish`, `cosmos_inference`,
   `image_read`, and the Cosmos 3 reasoner/generator/action paths.
2. **Recipes that remain pass untrusted free-text/paths via environment
   variables** (`C3_PROMPT`, `C3_GEN_OUT`, `C3_ACTION_INPUT`, …), read with
   `os.environ` inside the recipe — `just` does **not** interpolate env vars,
   so there is no template sink.
3. **Positionally-interpolated values are constrained**: enums
   (mode/strategy/preset/metric), numerics (coerced to `int`/`float`), or
   `validate_identifier()` (a strict metachar-free charset). Paths additionally
   go through workspace containment.
4. **Defense in depth**: `_common.just_run` rejects any argument containing
   shell/template metacharacters (`" ' \` ; | $( ${ && || > <` + newlines) at a
   single chokepoint.
5. **CI gate**: `tests/test_no_just_interpolation.py` parses every tool's AST
   and fails the build if any `just_run` positional argument is not provably
   safe (literal, sanitizer output, enum-guarded name, numeric, or a `*args`
   expansion whose backing tuple is itself all-safe).

### Capability confinement (defense in depth)

- **Filesystem (CWE-22)** — all input/output paths resolve symlinks, reject
  `..`/escape, and must stay within an allow-listed workspace root.
- **Network egress / SSRF (CWE-918)** — agent-influenced URLs are scheme- and
  host-allow-listed; private/link-local/loopback/metadata (e.g.
  `169.254.169.254`) ranges are blocked even when remote media is permitted.
- **NATS** — publish subjects must fall within an allow-listed namespace and
  may not contain wildcards or whitespace.

## Environment knobs

| Variable | Default | Purpose |
|----------|---------|---------|
| `COSMOS_WORKSPACE` | CWD (`+` tempdir) | `os.pathsep`-separated allow-listed roots for all file I/O. |
| `COSMOS_ALLOW_TEMP` | `1` | Include the system tempdir in the workspace (ffmpeg/frame output). Set `0` to fail-closed. |
| `COSMOS_URL_ALLOWLIST` | _(empty)_ | Comma-separated `host[:port]` entries the tools may POST to (localhost is always allowed). |
| `COSMOS_ALLOW_REMOTE_URLS` | off | `1` relaxes the host allow-list to any **public** host; private/metadata ranges stay blocked. |
| `COSMOS_NATS_NAMESPACE` | `cosmos,agent,perception` | Comma-separated allowed NATS subject prefixes. |

## Trusted-operator boundary

The `justfile` recipes are **also** usable directly by a human operator
(`just c3-reason`, `just c3-upsample`, `just post-train-*`). Those operator
paths route media references through the same hardened `_media_to_url`
resolver (workspace-confined, SSRF-guarded), so even the operator CLI cannot
read outside `COSMOS_WORKSPACE`. Run the tool executor as a least-privilege
user; do **not** start the vLLM server with `--allowed-local-media-path /`.

## Supply chain

- All GitHub Actions are pinned to full commit SHAs.
- PyPI publishing uses **Trusted Publishing (OIDC)** — no long-lived
  `PYPI_API_TOKEN`.
- Runtime dependencies carry upper version bounds and a committed `uv.lock`
  lockfile for reproducible installs.
