def simple_loss(prediction, noise):
    """Equation7.6: mean over samples of SUM over coordinates, not exact NLL."""
    return (prediction-noise).flatten(1).square().sum(1).mean()
