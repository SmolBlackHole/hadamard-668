# Research experiments

Only durable mathematical checks live here. Temporary solver experiments are
deleted after their result has been consolidated in `docs/research/`.

- `frame_integrability.py` verifies edge-word reconstruction and block
  integrability exhaustively for small `n` and against the tracker through
  `n=53`.

Run it from the repository root:

```powershell
python -m experiments.frame_integrability
```
