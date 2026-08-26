# ocr-id-parsing

### Repository structure

```text
ocr-id-parsing/
│
├── models/
│   ├── qwen3-vlm/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── chat_template.jinja
│   │   ├── processor_config.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   │
│   └── yolo/
│       └── best.onnx
│
├── scripts/
│   ├── cleanup.sh
│   ├── convert_data.py
│   └── generate_synthetic.sh
│
├── src/
│   ├── inference/
│   │   ├── infer.py
│   │   └── validate.py
│   │
│   └── training/
│       ├── detect_training.ipynb
│       └── vlm_fine_tuning.ipynb
│
├── .gitignore
├── PLAN.md
├── README.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```
