## Compile

From this directory, run:

```powershell
pdflatex autoencoder_visual_rl_report.tex
bibtex autoencoder_visual_rl_report
pdflatex autoencoder_visual_rl_report.tex
pdflatex autoencoder_visual_rl_report.tex
```

Alternatively, upload the complete `report/` directory to Overleaf and set
`autoencoder_visual_rl_report.tex` as the main document.

The paper includes the completed controlled benchmark for raw and latent PPO
across seeds 42, 123, and 456. Its final result figure is copied from
`results/main/comparison/eval_results_dashboard.png`.
