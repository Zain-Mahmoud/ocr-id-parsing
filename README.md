# ocr-id-parsing

### Repository structure

```
ocr-id-parsing/
├── data/
│   └── synthetic-ids/
│       ├── train/
│       │   ├── images/
│       │   └── IDLabels.csv
│       │
│       └── val/
│           ├── images/
│           └── IDLabels.csv
│
├── scripts/
│   ├── cleanup.sh                  # Remove generated datasets and artifacts
│   └── generate_synthetic.sh       # Generate the synthetic training dataset
│
├── src/
│   ├── inference/
│   │   ├── detect_infer.py         # YOLO field detection inference
│   │   ├── infer.py                # VLM inference pipeline
│   │   └── validate.py             # Model evaluation and validation
│   │
│   └── training/
│       ├── detect_training.ipynb   # YOLO detector training
│       └── vlm_fine_tuning.ipynb   # Vision-language model fine-tuning
│
├── .venv/
├── .env
├── .gitignore
├── PLAN.md
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```



Synthetic data generation instructions:
- To generate the fine-tuning synthetic data, run `./scripts/genarate_synthetic.sh type sample_size augmentation_batch_size` where `type` denotes whether to generate training or validation datasets and can either be `train` or `val`, `sample_size` is the number of unaugmented samples to generate and `augmentation_batch_size` is the number of augments to generate for each image. Note that this requires the [synthetic ID generator](https://github.com/Zain-Mahmoud/synthetic-id-generator) module to be installed.
