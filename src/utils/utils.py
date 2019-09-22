from .modules import *
from gym.wrappers import atari_preprocessing


class PreprocessImg(ObservationWrapper):
    def __init__(self, env, height=84, width=84, grayscale=True):
        super(PreprocessImg, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale

    def obs(self, img):
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), self.img_size,
                         interpolation=cv2.INTER_AREA)
        img = img / 255.0
        return img
