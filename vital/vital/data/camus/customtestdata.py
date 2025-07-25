import numpy as np
from dataclasses import dataclass, field
import h5py
import torch

class TestPatientData:
    def __init__(self):
        self.id = "test_patient"
        self.views = {
            "2CH": TestViewData(),
            "4CH": TestViewData(),
        }

@dataclass
class TestViewData:
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            group = f["patient_test"]["2CH"]
            
            # Load and convert to tensors
            self.img_proc = torch.tensor(group["img_proc"][...], dtype=torch.float32).unsqueeze(1)  # [N, 1, H, W]
            self.gt_proc = torch.tensor(group["gt_proc"][...], dtype=torch.long).unsqueeze(1)       # [N, 1, H, W]
            self.gt = torch.tensor(group["gt"][...], dtype=torch.long).unsqueeze(1)

        # Fill in placeholder data to mimic `ViewData`
        self.voxelspacing = [1.0, 1.0, 1.0, 1.0]  # dummy spacing unless stored
        self.instants = {'ED': 0, 'ES': 1} # dict
