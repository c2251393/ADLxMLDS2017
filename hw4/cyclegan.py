from utils import *
from data import *
from model import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', default='./data/',
                    help='data folder')
parser.add_argument('-l', '--lr', type=float, default=float(0.0002))
parser.add_argument('-b', '--batch_size', type=int, default=int(64))
parser.add_argument('-e', '--n_epoch', type=int, default=int(300))
parser.add_argument('-t', '--test_every', type=int, default=int(10))
parser.add_argument('--pre', action='store_true', help='load preprocessed file')

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class G(nn.Module):
    def __init__(self, conv_dim=64):
        super(G, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)

        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 3, 32, 32)
        return out

class D(nn.Module):
    """Discriminator for svhn."""
    def __init__(self, conv_dim=64):
        super(D, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*4, 4)
        n_out = 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out

args = parser.parse_args()
print("load the data in")
anime_data = AnimeRaw(args.data, args.pre)
celeb_data = Celeba(args.data, args.pre)
print("data loaded")

anime_loader = DataLoader(anime_data, batch_size=args.batch_size, shuffle=True)
celeb_loader = DataLoader(celeb_data, batch_size=args.batch_size, shuffle=True)

mG12 = G()
mG21 = G()
mD1 = D()
mD2 = D()

mG12.train()
mG21.train()
mD1.train()
mD2.train()

if USE_CUDA:
    mG12.cuda()
    mG21.cuda()
    mD1.cuda()
    mD2.cuda()

G_params = list(mG12.parameters()) + list(mG21.parameters())
D_params = list(mD1.parameters()) + list(mD2.parameters())

optG = optim.Adam(G_params, lr=args.lr, betas=(0.5, 0.999))
optD = optim.Adam(D_params, lr=args.lr, betas=(0.5, 0.999))
criterion = nn.CrossEntropyLoss()

iter = 0

def test_and_save(epoch):
    mG12.eval()
    mG21.eval()

    for i in range(10):
        anime_img = torch.from_numpy(random.choice(anime_data)).float().unsqueeze(0)
        gen_img = mG12(cu(Variable(anime_img))).data
        out_fn1 = os.path.join("output", "sample_%d_%d_anime.jpg" % (epoch, i))
        out_fn2 = os.path.join("output", "sample_%d_%d_celeb.jpg" % (epoch, i))
        to_img(anime_img[0], out_fn1)
        to_img(gen_img[0], out_fn2)

    model_name = "e%d.cyclegan.pt" % (epoch)
    torch.save(
        {
            "G12": mG12.state_dict(),
            "G21": mG21.state_dict(),
            "D1" : mD1.state_dict(),
            "D2" : mD2.state_dict()
        },
        os.path.join("models", model_name)
    )

def train(anime_batch, celeb_batch):
    d1l, d2l, g12l, g21l = 0, 0, 0, 0
    anime_batch = cu(Variable(anime_batch))
    celeb_batch = cu(Variable(celeb_batch))
    # train D
    loss = 0

    out = mD1(anime_batch)
    tmp = torch.mean((out - 1) ** 2)
    loss += tmp
    d1l += tmp.data[0]

    out = mD2(celeb_batch)
    tmp = torch.mean((out - 1) ** 2)
    loss += tmp
    d2l += tmp.data[0]

    loss = cu(loss)
    optD.zero_grad()
    loss.backward()
    optD.step()

    loss = 0

    fake_anime = mG21(celeb_batch)
    out = mD1(fake_anime)
    tmp = torch.mean(out ** 2)
    loss += tmp
    d1l += tmp.data[0]

    fake_celeb = mG12(anime_batch)
    out = mD2(fake_celeb)
    tmp = torch.mean(out ** 2)
    loss += tmp
    d2l += tmp.data[0]

    loss = cu(loss)
    optD.zero_grad()
    loss.backward()
    optD.step()

    # train G

    loss = 0
    fake_celeb = mG12(anime_batch)
    out = mD2(fake_celeb)
    recon_anime = mG21(fake_celeb)

    tmp = torch.mean((out - 1) ** 2) + torch.mean((anime_batch - recon_anime) ** 2)
    loss += tmp
    g12l += tmp.data[0]

    loss = cu(loss)
    optG.zero_grad()
    loss.backward()
    optG.step()

    loss = 0
    fake_anime = mG21(celeb_batch)
    out = mD1(fake_anime)
    recon_celeb = mG21(fake_anime)

    tmp = torch.mean((out - 1) ** 2) + torch.mean((celeb_batch - recon_celeb) ** 2)
    loss += tmp
    g21l += tmp.data[0]

    loss = cu(loss)
    optG.zero_grad()
    loss.backward()
    optG.step()

    return d1l, d2l, g12l, g21l


start = time.time()

for epoch in range(1, args.n_epoch+1):
    mG12.train()
    mG21.train()
    mD1.train()
    mD2.train()
    d1_loss, d2_loss, g12_loss, g21_loss = 0, 0, 0, 0
    for (i, (anime_batch, celeb_batch)) in enumerate(zip(anime_loader, celeb_loader)):
        d1l, d2l, g12l, g21l = train(anime_batch.float(), celeb_batch.float())
        d1_loss += d1l
        d2_loss += d2l
        g12_loss += g12l
        g21_loss += g21l
    print("Epoch %d %s d1 %g d2 %g g12 %g g21 %g" % (epoch, time_since(start),
                                                     d1_loss, d2_loss,
                                                     g12_loss, g21_loss))
    if epoch % args.test_every == 0:
        test_and_save(epoch)

