"""Convert the small sky segmentation model into a file the browser can use.

The model finds which pixels of a photo are sky. It is a small U-2-Net
("u2netp") trained by xiongzhu666 and shared under the MIT licence:
https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

It is only published in the ncnn format (a format for phones), so this script
reads the ncnn files and writes the same network as an ONNX file, which ONNX
Runtime Web can run in the browser. The weights are not changed.

Run from the project folder:  python tools/convert_sky_model.py
Needs:  pip install onnx==1.16.1 numpy

It downloads the two ncnn files from GitHub (a fixed version) and writes
space/sky.onnx.
"""

import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "space" / "sky.onnx"

# A fixed version of the source repository, so the result never changes by surprise
SOURCE = ("https://raw.githubusercontent.com/xiongzhu666/Sky-Segmentation-and-Post-processing/"
          "1f7811b32b64ddc957269defff84bc87a3f0b74f/")
PARAM_FILE = "skysegsmall_sim-opt-fp16.param"  # the network's layers, as text
WEIGHT_FILE = "skysegsmall_sim-opt-fp16.bin"   # the learned numbers

INPUT_SIZE = 384  # the network was built for 384 x 384 pixel images
OUTPUT_NAME = "1959"  # the final sky mask, as named in the ncnn files


def download(name):
    with urllib.request.urlopen(SOURCE + name) as response:
        return response.read()


class WeightReader:
    """Reads the learned numbers from the .bin file, one block after another."""

    def __init__(self, data):
        self.data = data
        self.position = 0

    def read_float32(self, count):
        values = np.frombuffer(self.data, dtype="<f4", count=count, offset=self.position)
        self.position += count * 4
        return values.copy()

    def read_tagged(self, count):
        # Each weight block starts with a 4-byte tag that says how it is stored
        tag = int.from_bytes(self.data[self.position:self.position + 4], "little")
        self.position += 4
        if tag == 0x01306B47:  # half precision (16-bit) numbers
            values = np.frombuffer(self.data, dtype="<f2", count=count, offset=self.position)
            self.position += (count * 2 + 3) // 4 * 4  # blocks are padded to 4 bytes
            return values.astype(np.float32)
        if tag == 0:  # plain 32-bit numbers
            return self.read_float32(count)
        raise ValueError(f"Unknown weight format {tag:#x}")


def parse_layers(param_text):
    """Turn each line of the .param file into (type, name, inputs, outputs, settings)."""
    lines = param_text.split("\n")
    assert lines[0].strip() == "7767517", "not an ncnn param file"
    layers = []
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        layer_type, name = parts[0], parts[1]
        n_in, n_out = int(parts[2]), int(parts[3])
        inputs = parts[4:4 + n_in]
        outputs = parts[4 + n_in:4 + n_in + n_out]
        settings = {}
        for item in parts[4 + n_in + n_out:]:
            key, value = item.split("=")
            settings[int(key)] = value
        layers.append((layer_type, name, inputs, outputs, settings))
    return layers


def convert(param_text, weight_data):
    weights = WeightReader(weight_data)
    nodes, initializers = [], []
    alias = {}  # ncnn "Split" layers only copy a value; we point the copies at the original

    def source(blob):
        return alias.get(blob, blob)

    def constant(name, array):
        initializers.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    for layer_type, name, inputs, outputs, s in parse_layers(param_text):
        inputs = [source(blob) for blob in inputs]

        if layer_type == "Input":
            input_name = outputs[0]

        elif layer_type == "Split":
            for blob in outputs:
                alias[blob] = inputs[0]

        elif layer_type == "Convolution":
            out_channels = int(s[0])
            kernel = int(s.get(1, 1))
            dilation = int(s.get(2, 1))
            stride = int(s.get(3, 1))
            pad = int(s.get(4, 0))
            assert 11 not in s and 12 not in s and 13 not in s and 14 not in s, "only square settings are handled"
            size = int(s[6])
            in_channels = size // (out_channels * kernel * kernel)
            w = weights.read_tagged(size).reshape(out_channels, in_channels, kernel, kernel)
            conv_inputs = [inputs[0], constant(name + ".weight", w)]
            if int(s.get(5, 0)):
                conv_inputs.append(constant(name + ".bias", weights.read_float32(out_channels)))
            activation = int(s.get(9, 0))
            conv_out = outputs[0] if activation == 0 else name + ".conv"
            nodes.append(helper.make_node(
                "Conv", conv_inputs, [conv_out], name=name,
                kernel_shape=[kernel, kernel], dilations=[dilation, dilation],
                strides=[stride, stride], pads=[pad] * 4))
            if activation == 1:
                nodes.append(helper.make_node("Relu", [conv_out], outputs, name=name + ".relu"))
            elif activation == 4:
                nodes.append(helper.make_node("Sigmoid", [conv_out], outputs, name=name + ".sigmoid"))
            elif activation != 0:
                raise ValueError(f"Unhandled activation {activation} in {name}")

        elif layer_type == "Pooling":
            assert int(s.get(0, 0)) == 0, "only max pooling is handled"
            kernel, stride = int(s[1]), int(s.get(2, 1))
            # ncnn's default padding mode rounds odd sizes up, like PyTorch's ceil_mode=True
            nodes.append(helper.make_node(
                "MaxPool", inputs, outputs, name=name,
                kernel_shape=[kernel, kernel], strides=[stride, stride], ceil_mode=1))

        elif layer_type == "Interp":
            assert int(s[0]) == 2 and int(s.get(6, 0)) == 0, "only bilinear resizing is handled"
            # Resize only height and width (axes 2 and 3) to the fixed size ncnn stored
            sizes = constant(name + ".sizes", np.array([int(s[3]), int(s[4])], dtype=np.int64))
            nodes.append(helper.make_node(
                "Resize", [inputs[0], "", "", sizes], outputs, name=name, axes=[2, 3],
                mode="linear", coordinate_transformation_mode="half_pixel"))

        elif layer_type == "Concat":
            assert int(s.get(0, 0)) == 0
            nodes.append(helper.make_node("Concat", inputs, outputs, axis=1, name=name))

        elif layer_type == "BinaryOp":
            assert int(s.get(0, 0)) == 0, "only addition is handled"
            nodes.append(helper.make_node("Add", inputs, outputs, name=name))

        elif layer_type == "Sigmoid":
            nodes.append(helper.make_node("Sigmoid", inputs, outputs, name=name))

        else:
            raise ValueError(f"Unhandled layer type {layer_type}")

    assert weights.position == len(weight_data), "not all weights were used"

    # Keep only the steps needed for the final mask (drops the 6 side outputs)
    needed, kept = {OUTPUT_NAME}, []
    for node in reversed(nodes):
        if any(out in needed for out in node.output):
            kept.append(node)
            needed.update(i for i in node.input if i)
    kept.reverse()
    used = {i for node in kept for i in node.input}
    initializers = [i for i in initializers if i.name in used]

    graph = helper.make_graph(
        kept, "sky_segmentation",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, INPUT_SIZE, INPUT_SIZE])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, 1, INPUT_SIZE, INPUT_SIZE])],
        initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="nephodea")
    model.ir_version = 8  # readable by the ONNX Runtime version the page uses
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    print("Downloading the ncnn model…")
    param_text = download(PARAM_FILE).decode()
    weight_data = download(WEIGHT_FILE)
    model = convert(param_text, weight_data)
    onnx.save(model, OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
