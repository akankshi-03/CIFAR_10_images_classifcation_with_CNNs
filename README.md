🚀 Project Overview

Built using TensorFlow / Keras.
Trains a CNN model to classify images into one of 10 categories.
Achieves good accuracy on the test dataset.
Includes model evaluation and visualization of results.

📂 Dataset

CIFAR-10 Dataset: Contains 10 classes — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
Dataset is automatically downloaded via Keras datasets API.

🛠️ Tech Stack

Language: Python
Frameworks & Libraries: TensorFlow/Keras, NumPy, Matplotlib, Seaborn
Notebook: Jupyter Notebook

📊 Model Architecture

Convolutional layers with ReLU activation
MaxPooling layers for dimensionality reduction
Dropout layers to prevent overfitting
Fully connected dense layers
Output layer with softmax activation for multi-class classification

Installation & Usage

Clone the repository:
git clone https://github.com/your-username/cifar10-cnn.git
cd cifar10-cnn

Install dependencies:
pip install -r requirements.txt

Run the Jupyter Notebook:
jupyter notebook CIFAR_10_Image_Classification_with_CNNs.ipynb

📈 Results

Training and validation accuracy are plotted during training.
Final model achieves ~X% test accuracy (update with your model’s accuracy).
Confusion matrix and classification report included.

📌 Future Improvements

Hyperparameter tuning
Data augmentation for better generalization
Try transfer learning with pretrained models like ResNet, VGG, or EfficientNet

👩‍💻 Author
Akankshi Dubey
