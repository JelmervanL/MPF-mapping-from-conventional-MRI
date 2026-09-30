# Pre-trained model

Put the downloaded pre-trained weights here as `latest_net_G.pth`. They belong to the proposed
model (`experiment=full_model`) and are loaded with `checkpoint=paper`:

```bash
uv run python test.py experiment=full_model test_set=umcu checkpoint=paper
```

The test results are written to this folder (`test_umcu/`, `test_mrrate/`).
