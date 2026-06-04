# Latency Benchmark

- Measurement: forward pass only
- Weights: randomly initialized architecture instances
- Purpose: architecture-level inference cost comparison under identical input length and batch size
- Batch size: `32`
- Warmup iterations: `10`
- Timed iterations: `50`

| Category | Model | Params | Device | Batch | ms/sample | ms/batch |
| --- | --- | --- | --- | --- | --- | --- |
| Proposed | Ours V1 | 158594 | cuda | 32 | 4.659 | 149.082 |
