"""Fit a rank-one update to a frozen two-feature model on synthetic CPU data."""

from __future__ import annotations


def run_experiment() -> dict[str, float | int | bool]:
    train = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    held_out = ((0.75, -0.25), (-0.2, 0.8), (0.4, 0.4), (-0.6, -0.6))
    frozen = (0.75, 0.25)
    target = (1.25, -0.5)
    adapter_a = [0.3, -0.2]
    adapter_b = 0.0
    steps = 100

    def error(x: tuple[float, float]) -> float:
        return sum((frozen[i] + adapter_b * adapter_a[i]) * x[i] for i in range(2)) - sum(
            target[i] * x[i] for i in range(2)
        )

    initial = sum(error(x) ** 2 for x in held_out) / len(held_out)
    for _ in range(steps):
        residuals = [(x, error(x)) for x in train]
        gradient_a = [
            sum(2 * residual * adapter_b * x[i] for x, residual in residuals) / len(train)
            for i in range(2)
        ]
        gradient_b = sum(
            2 * residual * sum(adapter_a[i] * x[i] for i in range(2)) for x, residual in residuals
        ) / len(train)
        adapter_a = [
            value - 0.2 * gradient for value, gradient in zip(adapter_a, gradient_a, strict=True)
        ]
        adapter_b -= 0.2 * gradient_b

    final = sum(error(x) ** 2 for x in held_out) / len(held_out)
    return {
        "rank": 1,
        "training_steps": steps,
        "baseline_mse": initial,
        "adapted_mse": final,
        "passed": final < initial * 0.05,
    }
