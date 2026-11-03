import cv2 as native_cv2
import numpy as np

from .raster import read, write, same_grid


class RasterCV:
    def __init__(self):
        self.profile = None
        self.valid = None

    def __getattr__(self, name):
        return getattr(native_cv2, name)

    def imread(self, path, flags=1):
        data, profile, valid = read(path)
        if self.profile is None:
            self.profile, self.valid = profile, valid
        else:
            same_grid(self.profile, profile)
            self.valid &= valid
        if flags == 0:
            return data[0].astype(np.uint8) if data.shape[0] == 1 else native_cv2.cvtColor(data[:3].transpose(1,2,0), native_cv2.COLOR_RGB2GRAY)
        if data.shape[0] == 1:
            return np.repeat(data[0,:,:,None],3,axis=2).astype(np.uint8)
        return data[:3].transpose(1,2,0)[:,:,::-1].astype(np.uint8)

    def imwrite(self, path, image, params=None):
        if self.profile is None:
            raise ValueError('Read the reference grid before writing')
        data = image[:,:,::-1].transpose(2,0,1) if image.ndim == 3 else image
        write(path, data, self.profile, self.valid)
        return True
