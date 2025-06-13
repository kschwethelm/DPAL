from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer, DistributedDPOptimizer
from opacus.data_loader import DPDataLoader
from opacus.accountants import create_accountant

import torch.distributed as dist

class DPSGD_PrivacyEngine:
    def __init__(
        self, 
        noise_multiplier, 
        delta, 
        max_grad_norm=1.0,
        num_groups=1,
        accountant="rdp"
    ):
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.num_groups = num_groups

        self.accountant = accountant
        self.accountants = [create_accountant(mechanism=self.accountant) for _ in range(num_groups)]

    def make_private(self, model, optimizer, data_loader, ddp=False, use_functorch=False):
        """ Use Opacus library to add DP-SGD functionalities """
        
        sample_rate = 1/len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        if use_functorch:
            world_size = dist.get_world_size() if ddp else 1
            Optim_cls = DistributedDPOptimizer if ddp else DPOptimizer
            optimizer = Optim_cls(
                optimizer=optimizer,
                noise_multiplier=self.noise_multiplier/self.max_grad_norm,
                max_grad_norm=self.max_grad_norm,
                expected_batch_size=expected_batch_size//world_size
            )

            data_loader = DPDataLoader.from_data_loader(data_loader, distributed=ddp)
        else:
            privacy_engine = PrivacyEngine()
            model, optimizer, data_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.noise_multiplier/self.max_grad_norm,
                max_grad_norm=self.max_grad_norm
            )

        return model, optimizer, data_loader

    def add_group(self):
        self.accountants.append(create_accountant(mechanism=self.accountant))
        self.num_groups += 1

    def add_steps(self, steps, sample_rate, group=0):
        assert group < self.num_groups, "Group index out of range."
        self.accountants[group].history.append([self.noise_multiplier, sample_rate, steps])

    def get_leakage(self, group=0):
        assert group < self.num_groups, "Group index out of range."
        return self.accountants[group].get_epsilon(delta=self.delta)