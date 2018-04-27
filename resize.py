#!/usr/bin/env python3


from glob import glob
from os import path
from sys import argv

from PIL import Image


if __name__ == '__main__':
    side, src, dst = int(argv[1]), argv[2], argv[3]

    for f in glob(path.join(src, '*.jpg')):
        img = Image.open(f)
        w, h = img.size
        smin = min(w, h)
        ow, oh = (w - smin) // 2, (h - smin) // 2
        square = img.crop((ow, oh, smin + ow, smin + oh))
        thumb = square.resize((side, side), Image.LANCZOS)

        thumb.save(path.join(dst, path.basename(f)))
