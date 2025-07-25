from typing import Dict
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.systems.computation import TrainValComputationMixin
from vital.utils.format.native import prefix

from crisp_uncertainty.evaluation.uncertainty.calibration import PixelCalibration, SampleCalibration
from crisp_uncertainty.evaluation.uncertainty.correlation import Correlation
from crisp_uncertainty.evaluation.uncertainty.mutual_information import UncertaintyErrorMutualInfo
from crisp_uncertainty.evaluation.data_struct import PatientResult, ViewResult
from crisp_uncertainty.utils.metrics import Dice
import numpy as np

class SegmentationComputationMixin(TrainValComputationMixin):
    """Mixin for segmentation train/val step.

    Implements generic segmentation train/val step and inference, assuming the following conditions:
        - the ``nn.Module`` used returns as single output the raw, unnormalized scores for each class in the predicted
          segmentation.
    The loss used is a weighted combination of Dice and cross-entropy.
    """

    # Fields to initialize in implementation of ``VitalSystem``
    #: Network called by ``SegmentationComputationMixin`` for test-time inference
    module: nn.Module

    def __init__(self, module: nn.Module, cross_entropy_weight: float = 0.1, dice_weight: float = 1, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            module: Module to train.
            cross_entropy_weight: Weight to give to the cross-entropy factor of the segmentation loss
            dice_weight: Weight to give to the cross-entropy factor of the segmentation loss
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore='module')
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")
        self.module = module
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight
        
        
        # Initialize your metrics here:
        #self.corr_metric = Correlation(Dice(labels=self.hparams.data_params.labels),upload_dir=Path("/home/zoayada1/intern_thesis_work/RMS_SampleFinder/depends/crisp/outputs/plots"))
        #self.pixel_calib = PixelCalibration()
        #self.sample_calib = SampleCalibration(accuracy_fn=Dice(labels=self.hparams.data_params.labels))
        #self.mi_metric = UncertaintyErrorMutualInfo()

    def forward(self, *args, **kwargs):  # noqa: D102
        return self.module(*args, **kwargs)
    
    def trainval_step(self, batch: Dict, batch_idx: int) -> Dict[str, Tensor]:
      # Detect if batch has 'views' attribute (test-like batch)
        if hasattr(batch, "views"):
            # Use a specific view or loop over views as needed; example uses '2CH'
            x = batch.views["2CH"].img_proc
            y = batch.views["2CH"].gt_proc
        else:
            # Usual train/val batch dict with keys Tags.img and Tags.gt
            x, y = batch[Tags.img], batch[Tags.gt]
    
        # Forward pass
        y_hat = self(x)
    
        # Compute loss (binary or multi-class)
        if y_hat.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.type_as(y_hat))
        else:
            ce = F.cross_entropy(y_hat, y)
    
        dice_values = self._dice(y_hat, y)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()
    
        loss = (self.cross_entropy_weight * ce) + (self.dice_weight * (1 - mean_dice))
    
        if self.is_val_step and batch_idx == 0:
            y_pred = y_hat.argmax(1) if y_hat.shape[1] > 1 else torch.sigmoid(y_hat).round()
            self.log_images(
                title="Sample",
                num_images=5,
                axes_content={
                    "Image": x.cpu().squeeze().numpy(),
                    "Gt": y.squeeze().cpu().numpy(),
                    "Pred": y_pred.detach().cpu().squeeze().numpy(),
                },
            )
    
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}

    def test_step(self, batch, batch_idx):
        # Run trainval_step but prefix metrics for testing
        result = prefix(self.trainval_step(batch, batch_idx), "test_")
    
        # Get batch size dynamically for logging
        if hasattr(batch, "views"):
            x = batch.views["2CH"].img_proc  # or choose a view dynamically
        else:
            x = batch[Tags.img]
    
        self.log_dict(result, batch_size=x.size(0), **self.val_log_kwargs)
        return result

