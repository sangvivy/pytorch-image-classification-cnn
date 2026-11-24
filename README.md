# PyTorch Image Classification Using a Residual CNN

A PyTorch-based image-classification project that classifies 32 × 32 RGB
images into ten object categories. The workflow includes a custom
PyTorch `Dataset`, training-time data augmentation, a convolutional
neural network with residual connections, hyperparameter tuning with
Optuna's Tree-structured Parzen Estimator (TPE), model evaluation, and
test-set prediction export.

## Project results

| Item | Result |
|---|---:|
| Labelled training images | 4,000 |
| Test images | 46,000 |
| Number of classes | 10 |
| Training/validation split | 80% / 20% |
| Training images used in split | 3,200 |
| Validation images | 800 |
| **Best validation accuracy** | **67.63%** |
| Test predictions generated | 46,000 |

**Note** 67.63% is the accuracy measured on the held-out
validation set. The test CSV does not contain labels, so test accuracy
could not be calculated.

## Classes

The model predicts one of the following categories:

1.  Airplane
2.  Automobile
3.  Bird
4.  Cat
5.  Deer
6.  Dog
7.  Frog
8.  Horse
9.  Ship
10. Truck

## Repository contents

``` text
pytorch-image-classification-cnn/
├── CAT_No_2_PyTorch_Image_Classification.ipynb
├── README.md
├── LICENSE
├── .gitignore
└── (downloaded dataset files and extracted data are kept locally)
```

The `dataset/` directory, ZIP archives, `.venv/`, and `Submission.csv`
are excluded by the current `.gitignore`. The generated submission file
is saved locally unless it is deliberately added to Git.

## Dataset

The assignment dataset is provided through [this Google Drive
folder](https://drive.google.com/drive/folders/12OeYCMZuqaxQFqzppItp9qlcGsH3dRZ0?usp=sharing).

The dataset contains: - `Train.csv`: image identifiers and their class
labels. - `Test.csv`: test image identifiers without labels. - ZIP
archive(s) containing the image files.

The images are RGB and have dimensions of 32 × 32 pixels. The labelled
dataset contains 4,000 images, with 400 images per class. The test set
contains 46,000 images.

### Expected local data arrangement

The notebook was developed with the extracted files arranged under the
project directory as follows:

``` text
ASS2/
├── dataset/
│   └── Trial/
│       ├── Train.csv
│       ├── Test.csv
│       └── train/
│           ├── <image files>
│           └── ...
├── CAT_No_2_PyTorch_Image_Classification.ipynb
└── ...
```

If the downloaded archive extracts into a different folder arrangement,
adjust the dataset paths in the notebook to match the actual extracted
locations. Keep the CSV image identifiers consistent with the image
filenames expected by the custom Dataset.

## Methodology

### 1. Data preparation

A custom class derived from `torch.utils.data.Dataset` loads the image
files and, for labelled examples, their corresponding class labels.
Images are converted to RGB and processed using separate transformations
for training and evaluation.

A stratified 80/20 split was used so that each class was represented
proportionally in both subsets: - Training subset: 3,200 images (320 per
class). - Validation subset: 800 images (80 per class).

### 2. Data augmentation

Training images use the following transformations: - Random crop to 32 ×
32 pixels with padding of 4 pixels. - Random horizontal flip with
probability 0.5. - Color jitter. - Conversion to tensors. -
Normalization.

Validation and test images use tensor conversion and normalization
without random augmentation. This keeps evaluation consistent.

### 3. Model architecture

The selected model is a compact convolutional neural network with
residual blocks.

At a high level, the architecture contains: - A convolutional stem that
maps the three RGB input channels to 32 feature channels. - Residual
blocks at 32, 64, and 128 channels. - Strided convolutions in selected
blocks to reduce spatial dimensions while increasing feature depth. -
Shortcut projections when the number of channels or spatial dimensions
changes. - Adaptive average pooling. - Dropout for regularization. - A
fully connected output layer with ten logits, one for each class.

The model is trained using cross-entropy loss with label smoothing
(`label_smoothing=0.05`).

### 4. Hyperparameter optimization

Optuna was used with its Tree-structured Parzen Estimator (TPE) sampler
to explore training configurations. The search used 10 trials, with five
epochs per trial to limit computational cost.

The selected configuration was:

| Hyperparameter | Selected value |
|---|---:|
| Learning rate | 0.00035747 |
| Weight decay | 0.00226482 |
| Dropout | 0.3330 |
| Batch size | 64 |
The best short-search trial reached 48.375% validation accuracy during
the five-epoch search. The selected configuration was subsequently
trained for a longer run.

### 5. Training and evaluation

The selected residual CNN was trained for 30 epochs. The best validation
accuracy was **67.63%** on the 800-image held-out validation set.

A classification report was generated to examine precision, recall, and
F1-score for each category.

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Airplane | 0.75 | 0.71 | 0.73 |
| Automobile | 0.82 | 0.81 | 0.82 |
| Bird | 0.53 | 0.54 | 0.53 |
| Cat | 0.54 | 0.47 | 0.50 |
| Deer | 0.60 | 0.61 | 0.60 |
| Dog | 0.59 | 0.69 | 0.64 |
| Frog | 0.75 | 0.68 | 0.71 |
| Horse | 0.67 | 0.70 | 0.69 |
| Ship | 0.75 | 0.76 | 0.76 |
| Truck | 0.77 | 0.79 | 0.78 |
| **Macro average** | **0.68** | **0.68** | **0.68** |
| **Weighted average** | **0.68** | **0.68** | **0.68** |

### 6. Test-set prediction

After model selection and validation, the selected architecture was
retrained using all 4,000 labelled images. The resulting model generated
predictions for all 46,000 test images.

The predictions were converted back to the required class labels and
exported to `Submission.csv`. The generated file contains 46,000 rows,
has no missing labels, and includes predictions from all ten classes.

Because the test set has no ground-truth labels, the test predictions
cannot be used to calculate test accuracy.

## Running the project

### Requirements

The project uses Python and PyTorch. The development environment used
CPU-only PyTorch 2.4.1. The notebook also uses common data-science and
image-processing libraries, including pandas, NumPy, scikit-learn,
Matplotlib, Pillow, and Optuna.

Create and activate a virtual environment from the project directory:

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the libraries used by the notebook:

``` powershell
python -m pip install --upgrade pip
pip install torch torchvision pandas numpy scikit-learn matplotlib pillow optuna jupyter
```

If you need to reproduce the original CPU environment, install the CPU
build of PyTorch using the installation command recommended by the
[official PyTorch installation
selector](https://pytorch.org/get-started/locally/).

### Download and extract the dataset

1.  Open the [assignment Google Drive
    folder](https://drive.google.com/drive/folders/12OeYCMZuqaxQFqzppItp9qlcGsH3dRZ0?usp=sharing).
2.  Download the supplied dataset archive(s).
3.  Extract the files into the project directory, following the expected
    arrangement shown above.
4.  Confirm that `Train.csv`, `Test.csv`, and the image directory are
    present.
5.  If your extracted folder names differ, update the path variables in
    the notebook.

### Run the notebook

Start Jupyter from the project directory:

``` powershell
jupyter notebook
```

Open `CAT_No_2_PyTorch_Image_Classification.ipynb` and run the cells in
order. Training from scratch can take time on a CPU. The notebook
contains the saved workflow and outputs from the completed run;
reproducing the training will execute the model-training cells again.

## Limitations

-   The labelled dataset contains 4,000 images, which limits the variety
    of examples available for learning.
-   Hyperparameter optimization was limited to 10 trials and five epochs
    per trial.
-   Evaluation used one stratified train-validation split; results may
    vary with another split.
-   CPU-only training constrained the number and size of experiments.
-   Performance varied across classes. In particular, cat and bird
    classification had lower F1-scores than several other categories.
-   Test accuracy is unknown because the test labels were not supplied.

## Possible improvements

-   Run a broader hyperparameter search with more trials and longer
    trial training.
-   Evaluate using repeated stratified cross-validation.
-   Explore additional augmentation strategies.
-   Compare the residual CNN with suitable pretrained architectures and
    transfer learning.
-   Use GPU acceleration to support more extensive experiments.
-   Investigate class-specific errors and use a confusion matrix to
    guide improvements.

## Reproducibility notes

-   The notebook is the primary source for the implementation and saved
    experimental outputs.
-   The dataset is hosted separately on Google Drive rather than
    committed to this repository.
-   The `.venv/` environment is local and is not part of the repository.
-   The current `Submission.csv` is generated locally and is ignored by
    Git unless deliberately force-added.

## License

MIT
