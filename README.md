# ocr-id-parsing

### Repository structure
```text
ocr-id-parsing/
├── src/
│   ├── data_generation/
│   │   ├── ...                      # Synthetic ID generation pipeline
│   │
│   ├── inference/
│   │   ├── detect_infer.py          # YOLO field detection inference
│   │   ├── infer.py                 # VLM inference pipeline
│   │   └── validate.py              # Evaluate model predictions
│   │
│   ├── training/
│   │   ├── detect_training.ipynb    # Train YOLO detector
│   │   └── vlm_fine_tuning.ipynb    # Fine-tune the VLM
│   │
│   └── index.js                     # Express.js backend
│
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
├── .venv/
├── .env
├── .gitignore
├── generate_synthetic.sh            # Generate synthetic training dataset
├── PLAN.md                          # Project planning notes
├── pyproject.toml                   # Project configuration and dependencies
├── requirements.txt                 # Python dependencies
├── uv.lock                          # uv lockfile
└── README.md
```



Instructions:
- To generate the fine-tuning synthetic data, run `./genarate_synthetic.sh type sample_size augmentation_batch_size` where `type` denotes whether to generate training or validation datasets and can either be `train` or `val`, `sample_size` is the number of unaugmented samples to generate and `augmentation_batch_size` is the number of augments to generate for each image.
