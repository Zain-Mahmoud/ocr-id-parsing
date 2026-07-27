# ocr-id-parsing

Instructions:
- To generate the fine-tuning synthetic data, run `./genarate_synthetic.sh type sample_size augmentation_batch_size` where `type` denotes whether to generate training or validation datasets and can either be `train` or `val`, `sample_size` is the number of unaugmented samples to generate and `augmentation_batch_size` is the number of augments to generate for each image.
