
class MaskConfig:

    def __init__(self, per_sample):
        self.per_sample = per_sample

    def __iter__(self):
        # Return the instance attributes as a dictionary for unpacking
        return iter(self.__dict__.items())

    def to_dict(self):
        # Return a dictionary of initialized attributes
        return {key: value for key, value in self.__dict__.items() if key in self.__class__.__init__.__code__.co_varnames and key != "per_sample"}

class PerlinNoiseConfig(MaskConfig):

    def __init__(self, 
                 scale=100, 
                 octaves=6, 
                 persistence=0.5, 
                 lacunarity=2.0,
                 threshold=0.5, 
                 sigma=1.0,
                 per_sample=10):
        super(PerlinNoiseConfig, self).__init__(per_sample=per_sample)
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.threshold = threshold
        self.sigma = sigma

class PresampledMaskConfig(MaskConfig):

    def __init__(self, directory, per_sample=10):
        super(PresampledMaskConfig, self).__init__(per_sample=per_sample)
        self.directory = directory

class MultiplePatchConfig(MaskConfig):

    def __init__(self):
        pass
