import numpy as np
import torch
from torchvision import transforms

class ImageUtils:

    _grayscaler = transforms.Grayscale()

    @staticmethod
    def permuteHWCtoCHW(img: np.ndarray) -> np.ndarray:
        """pytorch libraries expect format in CHW, many python libraries use HWC

        Args:
            img (np.ndarray): image in HWC format

        Returns:
            np.ndarray: image in CHW format
        """
        return np.transpose(img, (2, 0, 1))

    @staticmethod
    def permuteHWCtoCHWTensor(img: np.ndarray) -> torch.Tensor:
        imgNp = ImageUtils.permuteHWCtoCHW(img)
        return torch.tensor(imgNp.copy(), dtype=torch.float) # copy required because np.permute does not produce an array with contiguous memory
    
    @staticmethod
    def toTorchGray(img: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            img (torch.tensor): image in CHW format
        """
        return ImageUtils._grayscaler(img)
    
