import os
import argparse
import torch
import torchvision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default=None, help="Path to a dataset.")
    args = parser.parse_args()

    layer_map = {"conv1_1": "features.0", "conv1_2": "features.2", "conv2_1": "features.5", "conv2_2": "features.7", "conv3_1": "features.10", "conv3_2": "features.12", "conv3_3": "features.14", "conv4_1": "features.17", "conv4_2": "features.19", "conv4_3": "features.21", "conv5_1": "features.24", "conv5_2": "features.26", "conv5_3": "features.28", "fc6": "classifier.0", "fc7": "classifier.3", "fc8": "classifier.6"}

    model = torchvision.models.vgg16(num_classes=205)
    s = torch.load(args.infile)
    model.load_state_dict({replace(kn,):v for kn, v in s.items()})
    print(model)


def replace(key, mapping):
    k = key[:key.rfind(".")]
    return key.replace(k, mapping[k])

if __name__ == "__main__":
    main()
