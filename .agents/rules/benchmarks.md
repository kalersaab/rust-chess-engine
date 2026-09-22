# Benchmarking Guidelines

- Do **not** use fixed-time limits (e.g. `500ms`) as the primary development benchmark or evaluation metric.
- Use **fixed node counts** instead to ensure reproducible, machine-independent measurements.
- Standard benchmark tiers to use:
  - `10k` nodes (fast smoke tests)
  - `100k` nodes
  - `1M` nodes
  - `10M` nodes (deep evaluation / regression testing)
