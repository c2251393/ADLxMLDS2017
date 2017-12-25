from utils import *

# Face = namedtuple('Face', ('id', 'feat', 'tag'))
Face = namedtuple('Face', ('id', 'feat', 'eyes', 'hair'))

class Anime(Dataset):
    def __init__(self, dir, pre=False):
        self.id2eyes = {}
        self.id2hair = {}
        self.ids = []
        id_tag_csv = csv.reader(open(os.path.join(dir, 'tags_clean.csv')))
        for (i, (id, attrs)) in enumerate(id_tag_csv):
            if i >= 200:
                break
            attrs = attrs.split('\t')
            # print(id, attrs)
            self.id2eyes[id] = 0
            self.id2hair[id] = 0
            self.ids.append(id)
            for attr in attrs:
                attr = attr.split(':')[0]
                if attr in EYES:
                    self.id2eyes[id] = EYES.index(attr)
                if attr in HAIR:
                    self.id2hair[id] = HAIR.index(attr)
        # print(list(self.id2hair.items())[:10])
        self.data = []
        for id in tqdm(self.ids):
            if not pre:
                img_fn = os.path.join(dir, 'faces', '%s.jpg' % id)
                img = skimage.io.imread(img_fn)
                img = skimage.transform.resize(img, (64, 64)) # (64,64,3)
                img = np.transpose(img, (2, 0, 1))
                img_fn = os.path.join(dir, 'faces', '%s.npy' % id)
                img.tofile(open(img_fn, 'wb'))
            else:
                img_fn = os.path.join(dir, 'faces', '%s.npy' % id)
                img = np.fromfile(img_fn).reshape((3, 64, 64))
            eye = self.id2eyes[id]
            hair = self.id2hair[id]
            self.data.append(Face(id, img, EYES_VEC[eye], HAIR_VEC[hair]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
