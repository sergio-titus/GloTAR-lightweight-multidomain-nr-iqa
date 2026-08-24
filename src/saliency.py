
import cv2
import numpy as np


def saliency_region_proposals(
    image_rgb,
    max_regions=6,
    min_area_ratio=0.005,
    padding_ratio=0.10,
):
    img = np.asarray(image_rgb)

    if img.ndim != 3 or img.shape[2] != 3:
        return []

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    h, w = gray.shape

    sw = 64
    sh = max(
        32,
        int(round(h * sw / max(w, 1)))
    )

    small = cv2.resize(
        gray,
        (sw, sh),
        interpolation=cv2.INTER_AREA
    )

    fft = np.fft.fft2(small)

    amp = np.abs(fft)
    phase = np.angle(fft)

    log_amp = np.log(amp + 1e-8)

    residual = (
        log_amp
        - cv2.blur(
            log_amp.astype(np.float32),
            (3, 3)
        )
    )

    sal = np.abs(
        np.fft.ifft2(
            np.exp(
                residual + 1j * phase
            )
        )
    ) ** 2

    sal = cv2.GaussianBlur(
        sal.astype(np.float32),
        (5, 5),
        0
    )

    sal = cv2.resize(
        sal,
        (w, h)
    )

    sal -= sal.min()

    if sal.max() > 0:
        sal /= sal.max()

    sal8 = np.clip(
        sal * 255,
        0,
        255
    ).astype(np.uint8)

    _, binary = cv2.threshold(
        sal8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = float(h * w)
    candidates = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        if bw * bh < min_area_ratio * total_area:
            continue

        score = float(
            sal[y:y+bh, x:x+bw].mean()
        )

        px = int(padding_ratio * bw)
        py = int(padding_ratio * bh)

        candidates.append(
            (
                score,
                (
                    max(0, x-px),
                    max(0, y-py),
                    min(w, x+bw+px),
                    min(h, y+bh+py)
                )
            )
        )

    candidates.sort(
        key=lambda z: z[0],
        reverse=True
    )

    boxes = [
        box
        for _, box
        in candidates[:max_regions]
    ]

    if not boxes:
        boxes = [(0, 0, w, h)]

    return boxes
