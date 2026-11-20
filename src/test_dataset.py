from torch.utils.data import DataLoader
from dataset import PianoDataset

ds = PianoDataset()
print("samples per epoch:", len(ds))

x, y = ds[0]
print("one sample  cqt:", tuple(x.shape), " roll:", tuple(y.shape))
print("cqt range:", float(x.min()), "to", float(x.max()))
print("active key fraction:", float(y.mean()))

loader = DataLoader(ds, batch_size=4, shuffle=True)
xb, yb = next(iter(loader))
print("one batch   cqt:", tuple(xb.shape), " roll:", tuple(yb.shape))