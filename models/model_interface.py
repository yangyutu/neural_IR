import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl

import inspect
import importlib

class PLModelInterface(pl.LightningModule):

    def __init__(self, model_name, loss_name, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.load_loss()

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]


    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            #Model = getattr(importlib.import_module('.'+name, package=__package__), camel_name)
            Model = getattr(importlib.import_module('models.'+name), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def load_loss(self):
        name = self.hparams.loss_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            #Loss_CLS = getattr(importlib.import_module('..losses.'+name, package=__package__), camel_name)
            Loss_CLS = getattr(importlib.import_module('losses.'+name), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.loss_func = self.instancialize(Loss_CLS)

    def instancialize(self, object_cls, **other_args):
        """ Instancialize an object using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(object_cls.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return object_cls(**args1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PLModelTripletInterface")
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--loss_name", type=str, required=True)
        parser.add_argument("--lr", type=float, required=True)
        parser.add_argument("--lr_scheduler", type=str, default=None)

        return parent_parser

class PLModelTripletInterface(PLModelInterface):
    def __init__(self, model_name='bert_encoder', loss_name='triplet_loss', **kwargs):
        super().__init__(model_name, loss_name, **kwargs)

    def forward(self, input):
        embeddings = self.model(input)
        return embeddings
        
    def training_step(self, batch, batch_idx):
        token_anchor, token_pos, token_neg = batch
        embed_anchor = self.model(token_anchor)
        embed_pos = self.model(token_pos)
        embed_neg = self.model(token_neg)
        loss = self.loss_func((embed_anchor, embed_pos, embed_neg))
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

if __name__ == '__main__':
    model = PLModelTripletInterface(model_name='bert_encoder', loss_name='triplet', pretrained_model_name="distilbert-base-uncased")
