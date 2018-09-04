## Files to test the Owl implementation of Resnet50.

The pretrained weights from a Keras implementation have been taken from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5.

They are then converted to a format easily readable by OCaml using `save_weights.py` and then saved as an Owl network file (`resnet.network`) using `load_weights.ml`. You can also find the file `resnet.network` here https://drive.google.com/open?id=1dO8Ah2wBaXRgpW35t9wODmZp2ny5DfwX.

You can then run `test_resnet.ml` with the command `owl ./test.ml`, choosing the picture you want with the variable `src`.