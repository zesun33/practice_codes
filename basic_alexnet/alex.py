import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['AlexNet', 'alexnet']


class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0.0,
            Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            # self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        else:
            # self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.bn(x)
        # binAct = BinActive.apply
        # x = binAct(x)
        # x=BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x



class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            # nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1),
            nn.AvgPool2d(kernel_size=3, stride=2),
            BinConv2d(256, 384, kernel_size=3, stride=1, padding=1),
            BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1, dropout=.1),
            BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, dropout=.1),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            BinConv2d(256 * 6 * 6, 4096, Linear=True),
            BinConv2d(4096, 4096, Linear=True, dropout=0.1),
            # nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            # nn.Dropout(),
            nn.Linear(4096, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class AlexNet_flatten(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_flatten, self).__init__()
        self.num_classes = num_classes
        self.dr = 0.1
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            # nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dr),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dr),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dr),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'alexnet_best_51.pth.tar'
        # model_path = 'alexnet_XNOR_cpu.pth'
        pretrained_model = torch.load(model_path)
        model.features = torch.nn.DataParallel(model.features)
        model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model

class ConvReLU(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels, dropout=0.0,):
        super(ConvReLU, self).__init__()
        self.layer_type = 'ConvReLU'
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.dropout!=0:
            x = self.dropout(x)
        return x

class Vgg(nn.Module):

    def __init__(self, dr=0.1, num_classes=100):
        super(Vgg, self).__init__()
        self.dropout_ratio = dr
        self.features = nn.Sequential(
            ConvReLU(3, 64, dropout=0.1),           # ConvReLU(3,64):add(nn.Dropout(0.1))
            ConvReLU(64, 64),                       # ConvReLU(64,64)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2))
            #
            ConvReLU(64, 128, dropout=0.1),         # ConvReLU(64,128):add(nn.Dropout(0.1))
            ConvReLU(128, 128),                     # ConvReLU(128,128)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2))
            #
            ConvReLU(128, 256, dropout=0.1),        # ConvReLU(128,256):add(nn.Dropout(0.1))
            ConvReLU(256, 256, dropout=0.1),        # ConvReLU(256,256):add(nn.Dropout(0.1))
            ConvReLU(256, 256),                     # ConvReLU(256,256)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
            #
            ConvReLU(256, 512, dropout=0.1),        # ConvReLU(256,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512, dropout=0.1),        # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512),                     # ConvReLU(512,512)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
            #
            ConvReLU(512, 512, dropout=0.1),        # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512, dropout=0.1),        # ConvReLU(512,512):add(nn.Dropout(0.1))
            ConvReLU(512, 512),                     # ConvReLU(512,512)
            nn.AvgPool2d(kernel_size=2, stride=2),  # model:add(Avg(2,2,2,2):ceil())
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_ratio),         # classifier:add(nn.Dropout(0.1))
            nn.Linear(512, 4096,bias=False),# classifier:add(nn.Linear(512,512,false))
            nn.ReLU(True),                          # classifier:add(ReLU(true))
            nn.Dropout(self.dropout_ratio),         # classifier:add(nn.Dropout(0.1))
            nn.Linear(4096, 4096,bias=False),       # classifier:add(nn.Linear(512,10,false))
            nn.ReLU(True),                          # classifier:add(ReLU(true))
            nn.Dropout(self.dropout_ratio),         # additional dropout
            nn.Linear(4096, num_classes,bias=False),# additional dense layer
        )
        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)                    # model:add(nn.View(512))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

def vgg_net(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Vgg(**kwargs)
    if pretrained:
        model_path = 'model_best29.pth.tar'
        # model_path = 'alexnet_XNOR_cpu.pth'
        pretrained_model = torch.load(model_path)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in pretrained_model.items():
        #     name = k.replace(".module", "")  # remove `module.`
        #     new_state_dict[name] = v
        # load params
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        model.load_state_dict(pretrained_model['state_dict'])
        # model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model


class VGG_15_avga(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000, linea=512*7*7):
        super(VGG_15_avga, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(linea, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(4096, 4096, bias=False),  # Linear,
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4096, num_classes, bias=False)  # Linear,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15_avga(pretrained=False, dataset='cifar100' , **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        model = VGG_15_avga(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_15_avga(num_classes=100, linea=512,**kwargs)
    if pretrained:
        model_path = 'vgg15avg.pth.tar'
        print('loading pre-trained model from '+model_path)
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
        model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model
class VGG_15_avgb(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000, linea=512*7*7):
        super(VGG_15_avgb, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(linea, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(4096, 4096, bias=False),  # Linear,
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4096, num_classes, bias=False)  # Linear,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15_avgb(pretrained=False, dataset='cifar100' , **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        model = VGG_15_avgb(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_15_avgb(num_classes=100, linea=512,**kwargs)
    elif dataset == 'cifar10':
        model = VGG_15_avgb(num_classes=10, linea=512,**kwargs)
    if pretrained:
        model_path = 'vgg15avg.pth.tar'
        print('loading pre-trained model from '+model_path)
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
        model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model

class VGG_16(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000, linea=512*7*7):
        super(VGG_16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(linea, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, num_classes, bias=False)  # Linear,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_16(pretrained=False, dataset='cifar100' , **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        model = VGG_16(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_16(num_classes=100, linea=512,**kwargs)
    elif dataset == 'cifar10':
        model = VGG_16(num_classes=10, linea=512,**kwargs)
    if pretrained:
        model_path = 'vgg16.pth.tar'
        print('loading pre-trained model from '+model_path)
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
        model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model

class VGG_15_maxb(nn.Module):
    # def __init__(self,  dr=0.1, num_classes=1000):
    #     super(VGG_15_max, self).__init__()
    #     self.features = nn.Sequential(
    #         nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2)),
    #         nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2)),
    #         nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
    #         nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
    #     )
    #     self.classifier = nn.Sequential(
    #         nn.Dropout(0.1),
    #         nn.Linear(512, 4096, bias=False),  # Linear,
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         # nn.Linear(4096, 4096, bias=False),  # Linear,
    #         # nn.ReLU(),
    #         # nn.Dropout(0.1),
    #         nn.Linear(4096, 100, bias=False)  # Linear,
    #     )
    #
    #     self._initialize_weights()
    def __init__(self,  dr=0.1, num_classes=1000, linea=512*7*7):
        super(VGG_15_maxb, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReLU()

        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(linea, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(4096, 4096, bias=False),  # Linear,
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4096, num_classes, bias=False)  # Linear,
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15_maxb(pretrained=False,dataset='imagenet',**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        model = VGG_15_maxb(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_15_maxb(num_classes=100, linea=512,**kwargs)
    if pretrained:
        model_path = 'vgg15max.pth.tar'
        print('loading pre-trained model from '+model_path)
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
        model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model

class VGG_15_maxa(nn.Module):
    # def __init__(self,  dr=0.1, num_classes=1000):
    #     super(VGG_15_max, self).__init__()
    #     self.features = nn.Sequential(
    #         nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2)),
    #         nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2)),
    #         nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
    #         nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
    #
    #         nn.ReLU(),
    #         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
    #     )
    #     self.classifier = nn.Sequential(
    #         nn.Dropout(0.1),
    #         nn.Linear(512, 4096, bias=False),  # Linear,
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         # nn.Linear(4096, 4096, bias=False),  # Linear,
    #         # nn.ReLU(),
    #         # nn.Dropout(0.1),
    #         nn.Linear(4096, 100, bias=False)  # Linear,
    #     )
    #
    #     self._initialize_weights()
    def __init__(self,  dr=0.1, num_classes=1000, linea=512*7*7):
        super(VGG_15_maxa, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),

            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(linea, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(4096, 4096, bias=False),  # Linear,
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4096, num_classes, bias=False)  # Linear,
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15_maxa(pretrained=False,dataset='imagenet',**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        model = VGG_15_maxa(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_15_maxa(num_classes=100, linea=512,**kwargs)
    if pretrained:
        model_path = 'vgg15max.pth.tar'
        print('loading pre-trained model from '+model_path)
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
        model.load_state_dict(pretrained_model['state_dict'], strict=True)
    return model