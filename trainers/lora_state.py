"""Validate adapter-only state before mutating the model (no ML imports)."""


def validate_adapter_state(expected, supplied):
    if not isinstance(supplied, dict) or not supplied:
        raise ValueError("LoRA adapter must be a nonempty state dictionary.")
    missing = sorted(set(expected) - set(supplied))
    unexpected = sorted(set(supplied) - set(expected))
    if missing or unexpected:
        raise ValueError(f"LoRA keys mismatch: missing={missing}, unexpected={unexpected}")
    for key, reference in expected.items():
        if getattr(supplied[key], "shape", None) != reference.shape:
            raise ValueError(f"LoRA shape mismatch for {key}")
