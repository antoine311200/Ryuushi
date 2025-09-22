from ryuushi.resampling.resampler import ResamplingScheme, Resampler
from ryuushi.resampling.multinomial import MultinomialResampler

class ResamplerFactory:
    """Factory for creating resamplers"""
    @staticmethod
    def create(scheme: ResamplingScheme) -> Resampler:
        if scheme == ResamplingScheme.MULTINOMIAL:
            return MultinomialResampler()
        else:
            raise ValueError(f"Unknown resampling scheme: {scheme}")
