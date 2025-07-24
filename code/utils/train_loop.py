import os

import torch
import torch.distributed as dist

from torch.func import functional_call, vmap, grad
from opacus.utils.batch_memory_manager import BatchMemoryManager

from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MultilabelMatthewsCorrCoef
from utils.metrics import AverageMeter, accuracy
from utils.logging import ddp_logging

class Trainer():
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        train_loader,
        val_loader,
        data_name,
        num_classes,
        num_epochs,
        output_dir,
        privacy_engine,
        multi_label=False,
        max_physical_batch_size=1024,
        save_interval=-1,
        rank=None,
        ddp=False,
        use_functorch=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_name = data_name

        self.multi_label = multi_label
        num_classes = 5 if data_name=="chexpert" else num_classes
        self.mcc_index = MultilabelMatthewsCorrCoef(num_labels=num_classes) if multi_label else None
        self.num_epochs = num_epochs
        self.max_physical_batch_size = max_physical_batch_size

        self.save_interval = save_interval
        self.output_dir = output_dir

        self.privacy_engine = privacy_engine
        self.track_privacy = True
        self.steps = 0

        self.ddp = ddp
        self.world_size = dist.get_world_size() if ddp else 1
        self.rank = rank
        self.use_functorch = use_functorch

        if self.multi_label:
            self.metrics = {
                "loss": AverageMeter("Loss", ":.4e"),
                #"auc": AverageMeter("AUC", ":6.2f"),
                "val_loss": AverageMeter("Loss", ":.4e"),
                "val_auc": AverageMeter("AUC", ":6.2f"),
                "val_mcc": AverageMeter("MCC", ":6.2f")
            }
        else:
            self.metrics = {
                "loss": AverageMeter("Loss", ":.4e"),
                "acc": AverageMeter("Accuracy", ":6.2f"),
                #"auc": AverageMeter("AUC", ":6.2f"),
                "val_loss": AverageMeter("Loss", ":.4e"),
                "val_acc": AverageMeter("Accuracy", ":6.2f"),
                "val_auc": AverageMeter("AUC", ":6.2f")
            }

    def train(self, num_steps=None):
        # --------------------------------------------
        # for functorch
        # --------------------------------------------
        def compute_loss(params, buffers, sample, targets):
            sample = sample.unsqueeze(0)
            targets = targets.unsqueeze(0)

            output = functional_call(self.model, (params, buffers), (sample,))
            loss = self.criterion(output, targets)
            return loss, (loss, output[0])
        
        if self.use_functorch:
            compute_per_sample_grads = vmap(grad(compute_loss, has_aux=True), in_dims=(None, None, 0, 0), randomness="different")
        # --------------------------------------------

        self.model.to(self.device)
        local_steps = 0
        for epoch in range(self.num_epochs):
            self.metrics["loss"].reset()
            if not self.multi_label:
                self.metrics["acc"].reset()
            #self.metrics["auc"].reset()
            self.model.train()

            with BatchMemoryManager(
                    data_loader=self.train_loader, 
                    max_physical_batch_size=self.max_physical_batch_size, 
                    optimizer=self.optimizer
            ) as data_loader:
                for batch in data_loader:
                    skip_step = self.optimizer._check_skip_next_step(pop_next=False)
                        
                    if self.use_functorch:
                        # SNLI training not possible with functorch, thus, we skip this part
                        data, target = batch[0].to(self.device), batch[1].to(self.device)

                        params = {k: v.detach() for k, v in self.model.named_parameters()}
                        buffers = {k: v.detach() for k, v in self.model.named_buffers()}

                        per_sample_grads, (loss, output) = compute_per_sample_grads(params, buffers, data, target)

                        for n,p in self.model.named_parameters():
                            p.grad_sample = per_sample_grads[n] # opacus does the noising and clipping using this attribute

                        with torch.no_grad():
                            loss = loss.mean()
                    
                    else:
                        if self.data_name == "snli":
                            data = batch[0].to(self.device)
                            attention_mask = batch[1].to(self.device)
                            token_type_ids = batch[2].to(self.device)
                            target = batch[3].to(self.device)
                            outputs = self.model(data, attention_mask, token_type_ids, labels=target)
                            loss, output = outputs[:2]
                        else:
                            data, target = batch[0].to(self.device), batch[1].to(self.device)
                            output = self.model(data)
                            loss = self.criterion(output, target)

                        loss.backward()
                        
                    self.optimizer.step() # optimizer won't make a step unless logical batch is over
                    self.optimizer.zero_grad() # optimizer won't clear summed grad unless logical batch is over

                    with torch.no_grad():
                        if not self.multi_label:
                            acc1, = accuracy(output.view(-1, output.shape[-1]), target.view(-1), topk=(1,))
                            self.metrics["acc"].update(acc1.item(), data.size(0))
                        self.metrics["loss"].update(loss.item(), data.size(0))

                        # Skip virtual batches
                        if not skip_step: 
                            local_steps += 1
                            self.steps += 1
                            if num_steps is not None and self.steps >= num_steps:
                                break
                            
                            if self.ddp:
                                dist.reduce(loss, 0, op=dist.ReduceOp.SUM)

            # validation
            with torch.no_grad():
                ddp_logging("Validating...", self.rank)
                self.test()

                if self.ddp:
                    for met in self.metrics:
                        tensor = torch.tensor(self.metrics[met].avg).to(self.device)
                        dist.reduce(tensor, 0, op=dist.ReduceOp.SUM)
                        if self.rank==0:
                            self.metrics[met].avg = tensor.item() / self.world_size
                    
                log_str = ""
                for met in self.metrics:
                    log_str += f"{met}: {self.metrics[met].avg:.4f} - "
                ddp_logging(f"Epoch ({epoch+1}/{self.num_epochs}), Step {local_steps}, Global step {self.steps} - " + log_str, self.rank)

                if (self.save_interval > 0) and ((epoch+1) % self.save_interval) == 0:
                    ddp_logging.info("Saving model...", self.rank)
                    self.save(epoch+1)
                if num_steps is not None and self.steps >= num_steps:
                    break

        if (self.save_interval > 0) and (self.rank==0):
            ddp_logging.info("Saving model...", self.rank)
            self.save()

        if self.track_privacy and local_steps>0:
            for i in range(self.privacy_engine.num_groups):
                if self.privacy_engine.num_groups > 1:
                    sample_rate = self.train_loader.sample_rate[0] if i<self.privacy_engine.num_groups-1 else self.train_loader.sample_rate[-1] # Very ugly, but works for now
                else:
                    sample_rate = self.train_loader.sample_rate
                self.privacy_engine.add_steps(local_steps, sample_rate, group=i)
                eps = self.privacy_engine.get_leakage(group=i)
                ddp_logging(f"Privacy loss group {i}: eps = {eps}", rank=self.rank)

    @torch.no_grad()
    def test(self):
        self.metrics["val_loss"].reset()
        if not self.multi_label:
            self.metrics["val_acc"].reset()
        self.metrics["val_auc"].reset()

        model = self.model
        model.eval()

        outputs = []
        targets = []
        for i, (batch) in enumerate(self.val_loader):
            if self.data_name == "snli":
                data = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                token_type_ids = batch[2].to(self.device)
                target = batch[3].to(self.device)
                outputs_m = self.model(data, attention_mask, token_type_ids, labels=target)
                loss, output = outputs_m[:2]
            else:
                data, target = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(data)
                if self.data_name == "chexpert":
                    output = output[:, :5]
                loss = self.criterion(output, target)

            self.metrics["val_loss"].update(loss.item(), data.size(0))

            if self.multi_label:
                outputs.append(torch.sigmoid(output).cpu())
            else:   
                outputs.append(torch.softmax(output, dim=-1).cpu())
            targets.append(target.cpu())

            if self.multi_label:
                mcc = self.mcc_index(output.cpu(), target.type(torch.int32).cpu())
                self.metrics["val_mcc"].update(mcc.item(), data.size(0))
            else:
                acc1, = accuracy(output, target, topk=(1,))
                self.metrics["val_acc"].update(acc1.item(), data.size(0))

        outputs = torch.cat(outputs, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        if self.multi_label:
            auc1 = roc_auc_score(targets, outputs, average="macro")
        else:
            auc1 = roc_auc_score(targets, outputs, multi_class="ovr")
        self.metrics["val_auc"].update(auc1, outputs.shape[0])

    def save(self, epoch=-1):
        if epoch < 0:
            filename = "model_final.pth"
        else:
            filename = f"model_{epoch}.pth"

        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(self.output_dir, filename))
