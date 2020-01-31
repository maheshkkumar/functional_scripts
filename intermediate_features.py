import argparse
import os

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import vgg16
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PlaceHook(object):
    def __init__(self, module):
        self.register = module.register_forward_hook(self.place_hook)

    def place_hook(self, module, input, output):
        self.features = output
        self.register.remove()


class GetFeatures(object):
    def __init__(self, args):
        self.args = args

        # Note: other layers can be added to required_layers
        self.required_layers = [nn.Conv2d]

        self.data_transformer = transforms.Compose([transforms.ToTensor()])
        self.model = vgg16(pretrained=True)
        self.images = self.get_images()

    def get_images(self):
        """
        Method to read images, if input path is a directory
        :return:
        """
        images = []
        try:
            if os.path.isdir(args.data_path):
                images += [os.path.join(args.data_path, img) for img in os.listdir(args.data_path)]
            elif os.path.isfile(args.data_path):
                images.append(args.data_path)
            else:
                raise NotImplementedError
        except NotImplementedError:
            print("Incorrect data path, it should be either an image directory or an image path")

        return images

    def read_image(self, path):
        """
        Method to read image and apply data transformation
        """
        img = Image.open(path)
        img = img.convert('RGB') if img.mode == 'L' else img
        img = self.data_transformer(img)
        img = img.unsqueeze(0)

        return img

    def forward(self):
        self.model.to(device)
        self.model.eval()

        for img_path in tqdm(self.images, total=len(self.images)):
            img = self.read_image(img_path).to(device)
            img_name = os.path.split(img_path)[-1].split('.')[0]

            # fetching all the modules in the network and placing hooks on them
            modules = self.get_modules()
            model_features = [PlaceHook(m) for m in modules]

            with torch.no_grad():
                output = self.model(img)

            # saving retrieved intermediate features
            features = [f.features for f in model_features]
            torch.save({"features": features}, os.path.join(self.args.save_dir, img_name + '.pth'))

    def get_modules(self):
        """
        Method to get all the required modules
        """
        modules = []
        for name, module in self.model.named_modules():
            if type(module) in self.required_layers:
                modules.append(module)
        return modules


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', help="Path of the image/folder", required=True, type=str)
    parser.add_argument('-sd', '--save_dir', help='Path to save the features', required=True, type=str)

    args = parser.parse_args()
    gf = GetFeatures(args)
    gf.forward()
