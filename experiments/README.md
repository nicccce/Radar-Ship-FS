# Experiments Directory

本目录用于保存 `src/run_irfs.py` 自动生成的逐次运行产物，例如：

```text
experiments/<dataset>/seed-<seed>/selection.json
experiments/<dataset>/seed-<seed>/test.json
experiments/<dataset>/aggregate.json
```

运行产物通常较大且可复现，默认不提交到 Git。需要纳入论文或汇报的结果请整理到 `results/`。
