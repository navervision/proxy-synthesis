'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
from PIL import Image

def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))
