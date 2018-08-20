import h5py

fname = "resnet50_weights_tf_dim_ordering_tf_kernels.h5"
dfname = "resnet50_owl.hdf5"

f = h5py.File(fname, 'r')
data_file = h5py.File(dfname, 'w')

for node_name in f.keys():
    for param in f[node_name].keys():
        conv_weight = f[node_name][param].value.tolist()
        data_file.create_dataset(param, data=conv_weight)
      
f.close()  
data_file.close()
