This project is about using Nervous network to predict the XSS attack:

Need:
import numpy as np <br />
import pandas as pd <br />
from sklearn.model_selection import train_test_split<br />
import os<br />
import matplotlib.pyplot as plt<br />
import tensorflow as tf<br />
from tensorflow import keras<br />

Use:
python Model_learning.py <br />
Model: "sequential"<br />
_________________________________________________________________<br />
 Layer (type)                Output Shape              Param # <br />
=================================================================<br />
 conv2d (Conv2D)             (None, 8, 8, 128)         1280

 flatten (Flatten)           (None, 8192)              0

 dense (Dense)               (None, 128)               1048704

 dense_1 (Dense)             (None, 64)                8256

 dense_2 (Dense)             (None, 32)                2080

 dense_3 (Dense)             (None, 16)                528

 dense_4 (Dense)             (None, 8)                 136

 dense_5 (Dense)             (None, 1)                 9

=================================================================<br />
Total params: 1,060,993<br />
Trainable params: 1,060,993<br />
Non-trainable params: 0<br />
