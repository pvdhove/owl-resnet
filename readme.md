## Owl implementation of Resnet50.

The pretrained weights from a Keras implementation have been taken from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5.

They are then converted to a format easily readable by OCaml using `save_weights.py` and `load_weights.ml`. You can also find the network file `resnet.network` directly here https://drive.google.com/file/d/1IjVIHGFyJshCzIN7J-Hp9DUVsPapmdfL/view?usp=sharing.

You can then run `test.ml` with the command `owl ./test.ml`, choosing the picture you want with the variable `src`.