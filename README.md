## HASCII
Welcome to the HASCII dataset (Handwritten ASCII), a dataset containing 450 handwritten samples of each (writable)character in ASCII, for a total of 42,261 samples. Below you'll find information on the methods of collection and processing of the data, and human and reference model benchmarks.

Authors: Eric Higgins ([@ehiggins98](https://github.com/ehiggins98)) and Zak Kulphongpatana ([@zkulphong](https://github.com/zkulphong)).

**WARNING**: The checkpoint files are quite large, so if you're only interested in the dataset it might be a better idea to download that directly to avoid cloning.

### Data Collection
The alphanumeric characters in this dataset were sampled from the [EMNIST dataset](https://arxiv.org/abs/1702.05373) in equal numbers as the punctuation data.

The punctuation data was collected from around 225 undergraduate students. Each gave two samples of each character, for a total of around 450 samples per class. An approximately-equal number of samples per class were then sampled from the EMNIST By Class dataset to form the comprehensive dataset here.

### Processing
We performed roughly the same data processing as given in the [EMNIST dataset](https://arxiv.org/abs/1702.05373), which for completeness I'll outline here. This processing algorithm can be seen in [processor.py](https://github.com/ehiggins98/HASCII/blob/master/processing/processor.py).

Starting with scans of the format given in [sample scans](https://github.com/ehiggins98/HASCII/tree/master/sample%20scans), OpenCV was used to split each scan into a list of character samples. Some scans were somewhat faint, so the image was darkened by subtracting 20 from each RGB pixel value. The result was then converted to HSV and thresholded to isolate the character. A Gaussian Blur with standard deviation 1 was then applied to each image, as in EMNIST. Using OpenCV's findContours function, the minimal bounding box was found for each character, which was then centered in a black image with the width and height of the largest character and downscaled to [32, 32] using cubic interpolation.

## Format
The data is provided in dataset/all and dataset/punctuation for the entire dataset and the punctuation-only datasets, respectively. Each one contains several Numpy and TFRecord files, representing all data for that set, a training set, and a test set. The Numpy label files store an [n, 1] vector of class labels corresponding to the mappings in [labels.txt](https://github.com/ehiggins98/HASCII/labels.txt), and the image files contain an [n, 32, 32] matrix, where each [32, 32] slice represents an image (thus, the ith image is at images[i]). The TFRecord files contain the data in a format parseable by Tensorflow's Dataset API. See the [reference model input](https://github.com/ehiggins98/HASCII/blob/master/reference%20model/input.py) for the structure of this data. Character mappings are contained in [mappings.csv](https://github.com/ehiggins98/HASCII/blob/master/mappings.csv).

The sample count in each dataset is as follows:
* Complete set
    * Train: 38,061
    * Test: 4,200
* Punctuation-only set
    * Train: 12,899
    * Test: 1,400

### Human Benchmark
On the complete dataset I classified 500 examples manually, using the provided [human benchmarker](https://github.com/ehiggins98/HASCII/blob/master/human_benchmarker.py) script, and got 339 (67.8%) correct. On the punctuation-only dataset, I classified 300 examples and got 267 (89%) correct. These error rates are significantly
higher than one might expect, but with more thought it becomes reasonable. Many characters appear very similar, 
for example, '|', '\', '/', '1', 'I', and 'l'. Others are complicated by the preprocessing algorithm: for example,
an underscore and a hyphen are effectively indistinguishable.

### Reference Model
The reference model was trained on the train/test sets given in the dataset folders. In training, images were augmented by first cropping a random [28, 28] section of the image and padding it to [32, 32]. These were then scaled down randomly to a size between 27 and 32, and again padded to [32, 32]. Images were then rotated by an angle between 0 and 10 degrees, and normalized by scaling to the range [0, 1] and subtracting the mean of pixel values across the dataset. In evaluation, only the final normalization step is performed.

The model itself consists of a feature extractor with two convolutional layers, the first with 32 filters and the second with 64. Each has a 5x5 kernel, is padded to the same size, and has a ReLU activation applied afterward. A single DropConnect layer is applied after the feature extractor, again with a ReLU activation, followed by a softmax layer. An Adam optimizer with a maximum learning rate of 0.001 was used in training.

This model was trained on the complete and punctuation-only datasets with a batch size of 256 for 1200 and 800 iterations, respectively. It achieved 71.5% accuracy on the given test set for the complete dataset, and 92.6% on the punctuation-only dataset. The checkpoints can be found in the [reference model](https://github.com/ehiggins98/HASCII/blob/master/reference%20model) folder.