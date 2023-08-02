from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, init_lr0, init_lr2, end_learning_rate=0.0001, power=1.0, warmup_updates=2000):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        self.warmup_updates = warmup_updates
        if warmup_updates > 0:
            self.warmup_factor = 1.0 / warmup_updates
        else:
            self.warmup_factor = 1
        self.init_lr0 = init_lr0
        self.init_lr2 = init_lr2
        self.lr0 = 0
        self.lr2 = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    def get_plr(self, lr):
        lr_range = lr - self.end_learning_rate
        warmup = self.warmup_updates
        pct_remaining = 1 - (self.last_step - warmup) / (
                self.max_decay_steps - warmup
            )
        lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate
        return lr
    def get_lr(self):
        return [self.lr2]
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1        
        self.last_step = step if step != 0 else 1
        if self.warmup_updates > 0 and self.last_step <= self.warmup_updates:
            self.warmup_factor = self.last_step / float(self.warmup_updates)
            self.lr0 = self.warmup_factor * self.init_lr0
            self.lr2 = self.warmup_factor * self.init_lr2
        elif self.last_step <= self.max_decay_steps:
            self.lr0 = self.get_plr(self.init_lr0)
            self.lr2 = self.get_plr(self.init_lr2)
        for param_group in (self.optimizer.param_groups):
            if param_group['notchange']:
                param_group['lr'] = self.lr0
            else:
                param_group['lr'] = self.lr2
