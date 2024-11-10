# Plant Disease Classification - MALE_FinalProject - TEAM 03

This project uses TensorFlow and deep learning to classify plant diseases, specifically focusing on cassava leaf diseases.

## Contributions
- 23110019 - Huynh Gia Han
- 23110052 - Bui Tran Tan Phat
- 23110053 - Nguyen Nhat Phat
- 23110060 - Tran Huynh Xuan Thanh
## Features

- Utilizes the Cassava Leaf Disease dataset from TensorFlow Datasets
- Implements a convolutional neural network using MobileNetV3 as the base model
- Includes data preprocessing and augmentation
- Performs k-fold cross-validation for robust model evaluation
- Implements learning rate scheduling and early stopping
- Provides visualizations of model performance and predictions
- Includes a GUI for easy interaction with the trained model

## Requirements

See `requirements.txt` for a list of required Python packages.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/feedmybeast/MALE_FINALPROJECT.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```
python MALE_FinalProject.py
```

This will:
1. Load and preprocess the dataset
2. Perform k-fold cross-validation
3. Train the final model
4. Evaluate the model and display performance metrics
5. Open a GUI for classifying new images

## GUI Usage

1. Click the "Open Image" button in the GUI
2. Select an image file of a cassava leaf
3. The predicted disease class and confidence will be displayed

## Model Details

- Base model: MobileNetV3 (from TensorFlow Hub)
- Additional layers: Dense (128 units) with ReLU activation, Dropout (0.5), Dense (num_classes) with Softmax activation
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## Contact

For any questions or concerns, please open an issue on this repository.
Or email: 23110053@student.hcmute.edu.vn
