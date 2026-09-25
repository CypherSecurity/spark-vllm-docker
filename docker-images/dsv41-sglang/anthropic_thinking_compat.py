"""Build-time patch: make SGLang's Anthropic /v1/messages accept GLM/Z.ai-style thinking.

Z.ai clients (zcode) send {"thinking": {"type": "enabled"}} with no budget_tokens, which
upstream rejects with 400. The budget is never enforced by this backend (serving.py only
logs it), so:
  * enabled without budget_tokens  -> accepted
  * budget_tokens < 1024           -> raised to 1024
  * redacted_thinking in history   -> dropped (encrypted, unreadable locally) instead of 400
Fails the build if the upstream text it replaces is not found exactly once.
"""
from pathlib import Path

ROOT = Path("/sgl-workspace/sglang/python/sglang/srt/entrypoints/anthropic")

def sub(path, old, new):
    s = path.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{path.name}: expected 1 match, found {n}:\n{old}")
    path.write_text(s.replace(old, new))
    print(f"patched {path.name}")

sub(ROOT / "protocol.py",
'''            if self.budget_tokens is None:
                raise ValueError(
                    "thinking.budget_tokens is required when thinking.type is 'enabled'"
                )
            if self.budget_tokens < 1024:
                raise ValueError(
                    "thinking.budget_tokens must be >= 1024 (got {})".format(
                        self.budget_tokens
                    )
                )
''',
'''            # local patch (anthropic_thinking_compat): GLM/Z.ai clients send
            # {"type": "enabled"} without a budget; the backend never enforces it.
            if self.budget_tokens is not None and self.budget_tokens < 1024:
                self.budget_tokens = 1024
''')

sub(ROOT / "serving.py",
'''            if any(block.type == "redacted_thinking" for block in blocks):
                raise ValueError("Anthropic redacted_thinking history is not supported")
''',
'''            # local patch (anthropic_thinking_compat): drop unreadable encrypted blocks.
            blocks = [block for block in blocks if block.type != "redacted_thinking"]
''')
