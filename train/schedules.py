from torch.optim.lr_scheduler import LambdaLR
import math


def wsd_cosine_decay_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine decay with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)