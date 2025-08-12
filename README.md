# CNN Image Recognition Bootcamp 🖼️🤖

A comprehensive 5-day deep learning bootcamp focusing on Convolutional Neural Networks (CNNs) for image recognition tasks using popular datasets.

## 📋 Project Overview

This project explores image recognition using Convolutional Neural Networks across three different datasets:
- **MNIST**: Handwritten digit recognition
- **CIFAR-10**: Object classification (10 categories)
- **Cats vs Dogs**: Binary image classification

## 🗂️ Datasets

### 1. MNIST Dataset
- **Source**: Kaggle
- **Description**: 70,000 grayscale images of handwritten digits (0-9)
- **Image Size**: 28x28 pixels
- **Classes**: 10 (digits 0-9)

### 2. CIFAR-10 Dataset
- **Source**: Kaggle
- **Description**: 60,000 color images in 10 classes
- **Image Size**: 32x32 pixels
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### 3. Cats vs Dogs Dataset
- **Source**: Kaggle
- **Description**: Binary classification dataset with cat and dog images
- **Classes**: 2 (cats and dogs)

## 🛠️ Technologies & Libraries

- **Development Environment**: Google Colab
- **Deep Learning Framework**: TensorFlow/Keras
- **Data Visualization**: Matplotlib
- **Data Processing**: NumPy, Pandas

## 📊 Model Configuration

- **Training Epochs**: 5-10 epochs per model
- **Architecture**: Convolutional Neural Networks (CNN)
- **Optimization**: Various optimizers (Adam, SGD, etc.)
- **Loss Functions**: Categorical/Binary Crossentropy

## 📁 Project Structure

```
cnn-image-recognition-bootcamp/
├── notebooks/
│   ├── Day_1_MNIST_Recognition.ipynb
│   ├── Day_2_CIFAR10_Classification.ipynb
│   ├── Day_3_Cats_vs_Dogs.ipynb
│   ├── Day_4_Model_Comparison.ipynb
│   └── Day_5_Final_Project.ipynb
├── results/
│   ├── plots/
│   ├── model_metrics/
│   └── visualizations/
├── datasets/
│   └── (Downloaded directly in Google Colab)
└── README.md
```

## 🚀 Getting Started

### Access the Project

1. **Open Google Colab**
   - Navigate to [Google Colab](https://colab.research.google.com/)
   - Sign in with your Google account

2. **Load the notebooks**
   - Upload the .ipynb files to your Google Drive
   - Open notebooks directly in Colab
   - Or clone this repository and open notebooks from GitHub

3. **Dataset Access**
   - Datasets are downloaded directly within the Colab notebooks using Kaggle API
   - Ensure you have a Kaggle account for dataset access
   - Required datasets:
     - [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
     - [CIFAR-10 Dataset](https://www.kaggle.com/datasets/cifar10)
     - [Cats vs Dogs Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

### Running the Notebooks

- All notebooks are designed to run in Google Colab with GPU acceleration
- Each notebook includes necessary package installations
- Runtime typically takes 15-30 minutes per notebook depending on epochs

## 📈 Training Results

### Model Performance Summary

| Dataset | Epochs | Training Accuracy | Validation Accuracy | Loss |
|---------|--------|------------------|-------------------|------|
| MNIST | 5-10 | ~98% | ~97% | ~0.05 |
| CIFAR-10 | 5-10 | ~75% | ~70% | ~0.8 |
| Cats vs Dogs | 5-10 | ~85% | ~80% | ~0.4 |

## 📊 Visualizations

The project includes comprehensive visualizations using Matplotlib:
- Training/Validation accuracy curves
- Loss curves
- Sample predictions
- Confusion matrices
- Filter visualizations

## 💻 Usage

### Opening Individual Notebooks

1. Navigate to the `notebooks/` folder
2. Click on desired notebook (e.g., `Day_1_MNIST_Recognition.ipynb`)
3. Open with Google Colab
4. Run cells sequentially for complete workflow

### Notebook Overview

- **Day 1**: MNIST handwritten digit recognition
- **Day 2**: CIFAR-10 multi-class object classification  
- **Day 3**: Binary classification with Cats vs Dogs
- **Day 4**: Comparative analysis across all models
- **Day 5**: Final project integration and optimization

Each notebook includes:
- Data loading and preprocessing
- Model architecture design
- Training with 5-10 epochs
- Performance evaluation
- Visualization of results

## 📚 Learning Outcomes & Skills Developed

By the end of this 5-day bootcamp, participants will have developed expertise in:

### 🔧 Technical Skills
- **Image preprocessing and augmentation**: Data normalization, resizing, rotation, flipping, and other augmentation techniques
- **Deep learning fundamentals and CNN architecture**: Understanding convolutional layers, pooling, activation functions, and network design
- **Model training, evaluation, and optimization**: Hyperparameter tuning, loss functions, optimizers, and performance metrics
- **Visualization of results and metrics**: Creating insightful plots for training curves, confusion matrices, and model interpretability

### 💼 Professional Skills
- **Collaboration and documentation best practices**: Version control, code organization, and comprehensive project documentation
- **Showcasing ML projects for recruiters**: Portfolio development, technical presentation, and industry-relevant project structure

### 🎯 Core Competencies Gained
- End-to-end machine learning pipeline development
- Multi-dataset comparison and analysis
- Performance benchmarking across different image recognition tasks
- Industry-standard deep learning workflows

## 🎯 Key Features

- ✅ Implementation of CNN models for three different datasets
- ✅ Comprehensive data visualization and analysis
- ✅ Model performance comparison
- ✅ Well-documented Jupyter notebooks
- ✅ Modular code structure
- ✅ Training progress visualization

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for providing the datasets
- TensorFlow/Keras team for the excellent deep learning framework
- The deep learning community for tutorials and resources

## 🔗 Project Links

### 💻 Access Notebooks
- **Google Colab Notebooks**: https://colab.research.google.com/drive/1-X5PTN1GL4M_Ei6MxMIA740pBXUAtZmp?usp=sharing
