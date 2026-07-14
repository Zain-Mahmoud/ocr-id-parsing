### Dependencies
- `onnxruntime` for training output and loading during inference
- `opencv-python` for image preprocessing
- `ultralytics` for detecting card and orienting
- `easyocr` or `paddleocr` for field detection and recognition

### Post-training and validation
#### Steps
- Train and validate a yolo26n or yolo26s model (can tune as a hyperparameter) on training images to detect ID card location in image and draw bounding boxes. Use the train and vaildation options through the `yolo` api. optimize for best mAP50-95 (mean average precision at Intersection over Union threshold between 50-95 ) score.
- Use default MuSGD optimizer which is a hybrid between SGD updates and Muon-style updates
- Perform grid search to optimize the following hyperparameters (recommended by official ultralytics docs):
    - `imgsize`: size that all images get cropped to 
    - `lr0`: learning rate for optimizer
    - `batch`: batch size
    - `weight_decay`: L2 regularization term, penalizes larger weights to prevent overfitting
    - `epochs`: number of epochs (complete pass over the full dataset) to train data for

- Export best model to a `.onnx` file (instead of `.pt`) for inference stage 

- Validation using grid-search is often expensive and might not be necessary since we aren't training a model from scratch. Also, leaving out a held-out validation set reduce the training dataset (can be avoided by using cross-validation grid search cv). Depending on the training infrastructure and training dataset size, it might be wise to skip this step and use default hyperparameter values specified by ultralytics.

- Note: training/validation data argument is a path to a `.yaml` config file that specifies lots of configurations and values including the path to the training and validation data

#### Sample code
##### Without grid-search validation
```python
from ultralytics import YOLO

model = YOLO('yolo26n.pt') # load pre-trained model

results = model.train('/path/to/training/data') # train model with training data using default hyperparameter values

results.export('onnx') # export to onnx runtime file format
```

##### With grid-search validation
```python
from ultralytics import YOLO

grid = ... # define hyperparameters here

# initialize best score and model
best_map_score = -1 
best_model = None

for value in grid:
    model = YOLO('yolo26n.pt') # load pre-trained model

    results = model.train('/path/to/training/data') # train model with training data using default hyperparameter values

    metrics = results.val('/path/to/validation/data').map # get validation metrics on validation data
    curr_map = metrics.map
    ... # other validation metrics here

    if curr_map_score > best_map_score:
        #update best score and model
        best_map_score = curr_map_score
        best_model = results

best_model.export('onnx') # export best model to onnx runtime file format
```

#### TODOs and questions
- Look into configuration files
- Ask about GPU access and budget
- Ask about final platform for usecase
- Consult LLMs