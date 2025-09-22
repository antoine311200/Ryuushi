from ryuushi.resampling.resampler import ResamplingScheme, Resampler
from ryuushi.resampling.multinomial import MultinomialResampler
from ryuushi.resampling.systematic import SystematicResampler

class ResamplerFactory:
    """Factory for creating resamplers"""
    @staticmethod
    def create(scheme: ResamplingScheme) -> Resampler:
        if scheme == ResamplingScheme.MULTINOMIAL:
            return MultinomialResampler()
        elif scheme == ResamplingScheme.SYSTEMATIC:
            return SystematicResampler()
        else:
            raise ValueError(f"Unknown resampling scheme: {scheme}")
