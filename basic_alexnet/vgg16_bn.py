import torch
import os
import torch.nn as nn

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''

    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class VGG16_bn(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000):
        super(VGG16_bn, self).__init__()
        self.dr=dr
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(True),
            # nn.Dropout(dr),

            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            # nn.Dropout(dr),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            # nn.Dropout(dr),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),

            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),

            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            # nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),

            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=True),
            BinActive(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1),
            # nn.Dropout(dr),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Linear(512 * 7 * 7, 4096),  # Linear,
            nn.ReLU(True),
            nn.Dropout(dr),
            nn.Linear(4096, 1000)  # Linear,

        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg16_bn(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = vgg16_bn(**kwargs)
    if pretrained:
        model_path = 'vgg15_gpu.pth'
        # model_path = 'alexnet_XNOR_cpu.pth'
        pretrained_model = torch.load(model_path)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in pretrained_model.items():
        #     name = k.replace(".module", "")  # remove `module.`
        #     new_state_dict[name] = v
        # load params

        # model.load_state_dict(pretrained_model, strict=True)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        # torch.save(model.state_dict(), 'vgg15_gpu.pth')
        model.load_state_dict(pretrained_model, strict=True)
    return model