from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def identifier(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(int(value))
    return str(value)


def append_geometry(table, image_dir, output):
    path = Path(table)
    frame = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path)
    if frame.shape[1] != 17:
        raise ValueError('Expected ID + 15 features + label')
    compactness, extent = [], []
    for _, row in frame.iterrows():
        image_path = Path(image_dir) / (identifier(row.iloc[0]) + '.png')
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(str(row.iloc[0]))
        if not np.isin(image, [0, 255]).all():
            raise ValueError('Expected original black/white contour PNG, before resizing')
        contours, _ = cv2.findContours((image > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError('Empty contour image: ' + str(row.iloc[0]))
        contour = max(contours, key=cv2.contourArea)
        area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
        _, _, width, height = cv2.boundingRect(contour)
        if width != int(row.iloc[2]) or height != int(row.iloc[3]):
            raise ValueError('Contour PNG is cropped/resized or mismatched; regenerate features from source: ' + str(row.iloc[0]))
        compactness.append(4 * np.pi * area / perimeter**2 if perimeter else 0.0)
        extent.append(area / (width * height))
    frame.insert(len(frame.columns) - 1, 'Compactness', compactness)
    frame.insert(len(frame.columns) - 1, 'Extent', extent)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == '.xlsx':
        frame.to_excel(output, index=False)
    else:
        frame.to_csv(output, index=False, encoding='utf-8-sig')
    return frame
