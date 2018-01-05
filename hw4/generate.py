from utils import *
from data import *
from model import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('model', default='',
                    help='model file')
parser.add_argument('-test', default='data/sample_testing_text.txt',
                    help='testing text file')
parser.add_argument('-n', '--noise', type=int, default=int(100))

args = parser.parse_args()

mG = Generator(args.noise)
state_dict = torch.load(args.model, map_location=lambda storage, location: storage)
mG.load_state_dict(state_dict["gen"])
mG.eval()
if USE_CUDA:
    mG.cuda()

def gen(eyes, hair):
    # eyes, hair: (B, 50)
    batch_size = eyes.size(0)
    Z = cu(Variable(torch.zeros(batch_size, args.noise).normal_()))
    Y = torch.cat([cu(Variable(eyes)), cu(Variable(hair)), Z], dim=1)
    fX = mG(Y)
    return fX

def main():
    for line in open(args.test).readlines():
        print(line)
        id, desc = line.split(',')
        eyes, hair = get_tag(desc[:-1])
        for i in range(1, 5+1):
            out_fn = os.path.join("samples", "sample_%s_%d.jpg" % (id, i))
            y = gen(eyes, hair)
            print(out_fn)
            to_img(y[0].data, out_fn)

if __name__ == "__main__":
    main()
