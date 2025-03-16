import h5py
import numpy as np

old_weights_path = "test/test_best_weights.h5"        # <-- your old weights file
new_weights_path = "test/test_best_weights_converted.h5"  # new file to write

with h5py.File(old_weights_path, "r") as f_old, h5py.File(new_weights_path, "w") as f_new:
    # Copy all attributes of the root (if any) except 'layer_names' which we'll handle manually
    for key, val in f_old.attrs.items():
        if key != "layer_names":
            f_new.attrs[key] = val

    # The 'layer_names' attribute tells us which layer groups are in the file
    layer_names = f_old.attrs["layer_names"] if "layer_names" in f_old.attrs else []
    f_new.attrs.create("layer_names", layer_names, dtype=layer_names.dtype)

    for layer_name in layer_names:
        # Create a matching group in the new file
        g_old = f_old[layer_name]
        g_new = f_new.create_group(layer_name)

        # Copy layer attributes (which lists 'weight_names', etc.)
        for key, val in g_old.attrs.items():
            g_new.attrs[key] = val

        # For each weight tensor in this layer
        weight_names = g_old.attrs["weight_names"]
        g_new.attrs.create("weight_names", weight_names, dtype=weight_names.dtype)

        for weight_name in weight_names:
            w_old = g_old[weight_name][()]
            # Detect 4D conv kernels in format (out_ch, in_ch, kH, kW)
            if w_old.ndim == 4:
                # Usually old shape = (out_channels, in_channels, kH, kW)
                # We want new shape = (kH, kW, in_channels, out_channels)
                # That means transpose axes (0,1,2,3) -> (2,3,1,0)
                w_new = np.transpose(w_old, (2, 3, 1, 0))
            else:
                # For biases or 2D weights, 1D, etc., just copy as-is
                w_new = w_old

            g_new.create_dataset(weight_name, data=w_new)
