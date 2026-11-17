import cv2
import numpy as np
from typing import Optional

def apply_unsharp_mask(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0
) -> np.ndarray:
    """
    Classical edge-enhancing unsharp masking restoration baseline.
    sharp = original + amount * (original - blurred)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * image.astype(np.float32) - float(amount) * blurred.astype(np.float32)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    if threshold > 0:
        low_contrast_mask = np.abs(image.astype(np.int32) - blurred.astype(np.int32)) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened

def apply_wiener_filter(image: np.ndarray, kernel_size: int = 5, nsr: float = 0.05) -> np.ndarray:
    """
    Classical Wiener deconvolution filter approximation in frequency domain per color channel.
    """
    restored_channels = []
    # Gaussian point spread function (PSF) assumption
    psf = cv2.getGaussianKernel(kernel_size, -1)
    psf = psf @ psf.T
    psf /= psf.sum()

    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32) / 255.0
        h, w = channel.shape

        psf_padded = np.zeros((h, w), dtype=np.float32)
        kh, kw = psf.shape
        psf_padded[:kh, :kw] = psf
        # Roll PSF so center is at (0, 0)
        psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
        psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

        # FFT
        img_fft = np.fft.fft2(channel)
        psf_fft = np.fft.fft2(psf_padded)

        # Wiener filter H* / (|H|^2 + K)
        psf_conj = np.conj(psf_fft)
        wiener_filter = psf_conj / (np.abs(psf_fft) ** 2 + nsr)

        res_fft = img_fft * wiener_filter
        res = np.fft.ifft2(res_fft)
        res = np.real(res)
        res = np.clip(res * 255.0, 0, 255).astype(np.uint8)
        restored_channels.append(res)

    return cv2.merge(restored_channels)

def restore_classical(image: np.ndarray, method: str = "unsharp_mask", **kwargs) -> np.ndarray:
    """
    Dispatches chosen classical restoration algorithm.
    """
    if method == "wiener":
        return apply_wiener_filter(image, **kwargs)
    else:
        return apply_unsharp_mask(image, **kwargs)
