from utils import *

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.w1 = nn.Linear(TAG_DIM + self.noise_dim, 256 * 4 * 4)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv0 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 3, 5, stride=2, padding=2, output_padding=1)

        self.apply(weights_init)

    def forward(self, x):
        # (B, 200)
        x = self.w1(x)
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.bn0(x))
        # print(x.size())
        # (B, 4, 84, 84)
        x = F.relu(self.bn1(self.conv0(x)))
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = self.conv3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, use_bn):
        super(Discriminator, self).__init__()
        self.use_bn = use_bn
        self.conv0 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.conv1 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 5, stride=2, padding=2)

        self.conv4 = nn.Conv2d(512, 128, 1, stride=1)

        self.wtag = nn.Linear(TAG_DIM, 256)
        self.w1 = nn.Linear(2048, 2)

        if self.use_bn:
            self.bn0 = nn.BatchNorm2d(32)
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(128)

        self.apply(weights_init)

    def forward(self, x, tag):
        # x: (B, 3, 64, 64)
        # tag: (B, 100)
        tag = self.wtag(tag)
        # tag: (B, 256)
        if not self.use_bn:
            x = F.leaky_relu(self.conv0(x))
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            tag = tag.unsqueeze(2).unsqueeze(3)
            tag = tag.repeat(1, 1, x.size(2), x.size(3))
            cond = torch.cat([x, tag], dim=1)
            # cond: (B, 512, 4, 4)
            cond = F.leaky_relu(self.conv4(cond))
            cond = cond.view(cond.size(0), -1)
            x = self.w1(cond)
        else:
            x = F.leaky_relu(self.bn0(self.conv0(x)))
            x = F.leaky_relu(self.bn1(self.conv1(x)))
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = F.leaky_relu(self.bn3(self.conv3(x)))
            tag = tag.unsqueeze(2).unsqueeze(3)
            tag = tag.repeat(1, 1, x.size(2), x.size(3))
            cond = torch.cat([x, tag], dim=1)
            # cond: (B, 512, 4, 4)
            cond = F.leaky_relu(self.bn4(self.conv4(cond)))
            cond = cond.view(cond.size(0), -1)
            x = self.w1(cond)
        return x


if __name__ == '__main__':
    # x = Variable(torch.randn(64, 3, 64, 64))
    # tag = Variable(torch.randn(64, 100))
    # md = Discriminator(False)
    # md.eval()
    # print(md(x, tag))
    # print(md)
    x = Variable(torch.randn(TAG_DIM + 100).view(1, -1))
    md = Generator(100)
    md.eval()
    y = md(x)
    print(y)
    to_img(y, "test.png")
