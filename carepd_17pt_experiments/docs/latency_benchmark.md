# Latency Benchmark

- Measurement: forward pass only
- Weights: randomly initialized architecture instances
- Purpose: architecture-level inference cost comparison under identical input length and batch size
- Batch size: `32`
- Warmup iterations: `20`
- Timed iterations: `100`

| Category | Model | Params | Device | Batch | ms/sample | ms/batch |
| --- | --- | --- | --- | --- | --- | --- |
| Deep Learning | Temporal CNN | 188929 | cuda | 32 | 0.020 | 0.642 |
| Proposed | Ours V1 | 158594 | cuda | 32 | 0.242 | 7.749 |
| SOTA | Lu official | 147908 | cuda | 32 | 0.335 | 10.706 |
| SOTA | ST-GCN | 252097 | cuda | 32 | 0.522 | 16.708 |
| SOTA | MotionAGFormer-XS pretrained | 2307324 | cuda | 32 | 1.937 | 61.976 |
| SOTA | MotionBERT-Lite | 10814222 | cuda | 32 | 16.481 | 527.407 |
