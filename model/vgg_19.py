import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias= False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def vgg_block(in_channels, out_channels, num_layers, *args, **kwargs):
    layers = []
    for layer in range(num_layers):
        layers.append(conv_block(in_channels, out_channels)) if layer == 0 else layers.append(conv_block(out_channels, out_channels))
    return nn.Sequential(*layers)

class VGG_19(nn.Module):
    def __init__(self, in_channels, out_channels, enc_size, list_layers):
        super(VGG_19, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = out_channels
        self.enc_size = enc_size
        self.list_layers = list_layers

        def arch_list(enc_size, list_layers):
            layers = []
            for in_c, out_c, num_layer in zip(enc_size, enc_size[1:] + [enc_size[-1]], list_layers):
                layers.append(vgg_block(in_c, out_c, num_layer))
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.encoder = arch_list(self.enc_size, self.list_layers)

        self.decoder = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.out_channels),
            nn.Softmax(dim= 1)
        )
    
    def forward(self, inp):
        x = self.encoder(inp)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x
    
if __name__ == "__main__":
    model = VGG_19(3, 400, [3, 64, 128, 256, 512], [2, 2, 4, 4, 4])
    exm = torch.rand(6, 3, 224, 224)
    print(model)
    output = model(exm)
    print(output)
    print(output.shape)
    print(torch.argmax(output, dim = 1))
    print(output.argmax(dim = 1))