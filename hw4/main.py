from utils import *
from data import *
from model import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', default='./data/',
                    help='data folder')
parser.add_argument('-l', '--lr', type=float, default=float(0.0002))
parser.add_argument('-b', '--batch_size', type=int, default=int(64))
parser.add_argument('-e', '--n_epoch', type=int, default=int(300))
parser.add_argument('-d', '--D_step', type=int, default=int(1))
parser.add_argument('-g', '--G_step', type=int, default=int(2))
parser.add_argument('-t', '--test_every', type=int, default=int(10))
parser.add_argument('-n', '--noise', type=int, default=int(100))
parser.add_argument('--pre', action='store_true', help='load preprocessed file')

args = parser.parse_args()
print("load the data in")
data = Anime(args.data, args.pre)
print("data loaded")
# try:
    # data = pickle.load(open('data.pt', 'rb'))
# except:
    # print("load the data in")
    # data = Anime(args.data)
    # print("data loaded")
    # pickle.dump(data, open('data.pt', 'wb'))

loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
fake_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

mG = Generator(args.noise)
mD = Discriminator(True)

mG.train()
mD.train()

if USE_CUDA:
    mG.cuda()
    mD.cuda()

optG = optim.Adam(mG.parameters(), lr=args.lr, betas=(0.5, 0.999))
optD = optim.Adam(mD.parameters(), lr=args.lr, betas=(0.5, 0.999))
criterion = nn.CrossEntropyLoss()

def round(iter):
    if iter % (args.G_step + args.D_step) < args.D_step:
        return 'D'
    else:
        return 'G'

iter = 0

fixed_noise = [cu(Variable(torch.zeros(1, args.noise).normal_())) for i in range(5)]

def gen(eyes, hair, Z=None):
    # eyes, hair: (B, 50)
    batch_size = eyes.size(0)
    if Z is None:
        Z = cu(Variable(torch.zeros(batch_size, args.noise).normal_()))
    Y = torch.cat([cu(Variable(eyes)), cu(Variable(hair)), Z], dim=1)
    fX = mG(Y)
    return fX

def train_D(batch, fake_batch):
    ids, feats, eyes, hair = batch
    batch_size = eyes.size(0)

    wids, wfeats, weyes, whair = fake_batch
    # print(ids[:5], wids[:5])

    X = cu(Variable(feats.float()))
    wX = cu(Variable(wfeats.float()))
    fX = gen(eyes, hair)

    H = torch.cat([cu(Variable(eyes)), cu(Variable(hair))], dim=1)
    wH = torch.cat([cu(Variable(weyes)), cu(Variable(whair))], dim=1)

    sr = mD(X, H) # real img right txt

    sw = mD(X, wH) # real img wrong txt
    sf = mD(fX, H) # fake img right txt
    sW = mD(wX, H) # wrong img right txt

    ones = cu(Variable(torch.ones(batch_size).long()))
    zeros = cu(Variable(torch.zeros(batch_size).long()))

    loss = criterion(sr, ones) + \
            (criterion(sw, zeros) + criterion(sf, zeros) + criterion(sW, zeros)) / 3
    loss = cu(loss)

    optD.zero_grad()
    loss.backward()
    optD.step()

    return loss.data[0]

def train_G(batch):
    ids, feats, eyes, hair = batch
    batch_size = eyes.size(0)
    X = cu(Variable(feats.float()))
    fX = gen(eyes, hair)

    H = torch.cat([cu(Variable(eyes)), cu(Variable(hair))], dim=1)

    sf = mD(fX, H) # fake image right text
    ones = cu(Variable(torch.ones(batch_size).long()))
    # zeros = cu(Variable(torch.zeros(batch_size).long()))

    loss = criterion(sf, ones)
    loss = cu(loss)

    optG.zero_grad()
    loss.backward()
    optG.step()

    return loss.data[0]

def test_and_save(epoch):
    mD.eval()
    mG.eval()
    test_fn = os.path.join("data", "sample_testing_text.txt")
    for line in open(test_fn).readlines():
        print(line)
        id, desc = line.split(',')
        eyes, hair = get_tag(desc[:-1])
        for i in range(1, 5+1):
            out_fn = os.path.join("output", "sample_%d_%s_%d.jpg" % (epoch, id, i))
            y = gen(eyes, hair, fixed_noise[i-1])
            print(out_fn)
            to_img(y[0].data, out_fn)

    model_name = "D%d.G%d.n%d.e%d.pt" % (args.D_step, args.G_step, args.noise, epoch)
    torch.save(
        {
            "gen": mG.state_dict(),
            "disc": mD.state_dict()
        },
        os.path.join("models", model_name)
    )

start = time.time()

for epoch in range(1, args.n_epoch+1):
    mD.train()
    mG.train()
    d_loss, g_loss = 0, 0
    for (i, (batch, fake_batch)) in enumerate(zip(loader, fake_loader)):
        # print(gen(batch[2], batch[3]))
        # break
        if round(iter) == 'D':
            d_loss += train_D(batch, fake_batch)
        else:
            g_loss += train_G(batch)
        iter += 1
    print("Epoch %d %s d %g g%g" % (epoch, time_since(start), d_loss, g_loss))
    if epoch % args.test_every == 0:
        test_and_save(epoch)
