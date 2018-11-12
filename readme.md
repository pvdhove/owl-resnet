## Owl implementation of Resnet50.

The pretrained weights from a Keras implementation have been taken from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5.

You can find the weight file `resnet.weights` directly here https://drive.google.com/open?id=1Iy6yCpYcXa3BhXsdOjO-DgnzhE3u18ht (they have been converted with `save_weights.py` and `load_weights.ml`) and place it at the source of the directory.

You can then run `test.ml` with the command `owl ./test.ml`, choosing the picture you want with the variable `src`.