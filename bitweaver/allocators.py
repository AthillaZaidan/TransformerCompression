from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Mapping, Sequence


LayerOptions = Sequence[Mapping[int, Mapping[str, float]]]


@dataclass(frozen=True)
class AllocationResult:
    method: str
    config: list[int]
    total_size: int
    total_loss: float
    feasible: bool = True


def _score(options: LayerOptions, config: Sequence[int], method: str, budget: int | None = None) -> AllocationResult:
    total_size = int(sum(options[i][bit]["size"] for i, bit in enumerate(config)))
    total_loss = float(sum(options[i][bit]["loss"] for i, bit in enumerate(config)))
    feasible = budget is None or total_size <= budget
    return AllocationResult(method, list(config), total_size, round(total_loss, 10), feasible)


def dp_allocate(options: LayerOptions, budget: int) -> AllocationResult:
    n = len(options)
    dp = [[inf] * (budget + 1) for _ in range(n + 1)]
    parent: list[list[int | None]] = [[None] * (budget + 1) for _ in range(n + 1)]
    prev_budget: list[list[int | None]] = [[None] * (budget + 1) for _ in range(n + 1)]

    for b in range(budget + 1):
        dp[0][b] = 0.0

    for i, layer in enumerate(options, start=1):
        for b in range(budget + 1):
            for bit, values in layer.items():
                size = int(values["size"])
                if size > b:
                    continue
                candidate = dp[i - 1][b - size] + float(values["loss"])
                if candidate < dp[i][b]:
                    dp[i][b] = candidate
                    parent[i][b] = bit
                    prev_budget[i][b] = b - size

    best_budget = min(range(budget + 1), key=lambda b: dp[n][b])
    if dp[n][best_budget] == inf:
        return AllocationResult("DP", [], 0, inf, False)

    config: list[int] = []
    b = best_budget
    for i in range(n, 0, -1):
        bit = parent[i][b]
        if bit is None:
            return AllocationResult("DP", [], 0, inf, False)
        config.append(bit)
        previous = prev_budget[i][b]
        if previous is None:
            return AllocationResult("DP", [], 0, inf, False)
        b = previous
    config.reverse()
    return _score(options, config, "DP", budget)


def uniform_allocate(options: LayerOptions, bit_width: int, budget: int | None = None) -> AllocationResult:
    return _score(options, [bit_width] * len(options), f"Uniform {bit_width}-bit", budget)


def greedy_allocate(options: LayerOptions, budget: int) -> AllocationResult:
    config = [max(layer) for layer in options]
    current = _score(options, config, "Greedy")

    while current.total_size > budget:
        best_move: tuple[float, int, int] | None = None
        for i, layer in enumerate(options):
            current_bit = config[i]
            lower_bits = [bit for bit in layer if bit < current_bit]
            if not lower_bits:
                continue
            next_bit = max(lower_bits)
            saved = layer[current_bit]["size"] - layer[next_bit]["size"]
            loss_added = layer[next_bit]["loss"] - layer[current_bit]["loss"]
            ratio = loss_added / saved if saved else inf
            candidate = (ratio, i, next_bit)
            if best_move is None or candidate < best_move:
                best_move = candidate

        if best_move is None:
            break
        _, layer_index, bit = best_move
        config[layer_index] = bit
        current = _score(options, config, "Greedy")

    return _score(options, config, "Greedy", budget)
