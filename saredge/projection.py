import numpy as np
import pandas as pd

from .raster import read, write


def project(predictions, candidate_ids, output):
    frame = pd.read_csv(predictions) if not isinstance(predictions, pd.DataFrame) else predictions
    data, profile, valid = read(candidate_ids)
    labels = data[0]
    if frame['id'].duplicated().any() or not frame['prediction'].isin([0,1]).all():
        raise ValueError('Expected unique contour IDs and binary predictions')
    available = np.unique(labels[labels > 0])
    if not np.isin(frame['id'].to_numpy(), available).all():
        raise ValueError('Prediction IDs do not belong to this candidate raster')
    if not np.isin(available,frame['id'].to_numpy()).all():
        raise ValueError('Some candidate contours lack predictions')
    selected = frame.loc[frame['prediction']==1,'id'].to_numpy()
    mask = np.isin(labels,selected) & valid
    rgb = np.zeros((3,*labels.shape),np.uint8)
    rgb[0,mask] = 255
    write(output,rgb,profile,valid)
    return int(mask.sum())
