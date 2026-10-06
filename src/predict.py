"""Inference on a Figshare .mat file, with no ground-truth access."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from src.models import TumorNet


@torch.no_grad()
def predict(checkpoint, source, output):
    torch.set_num_threads(4)
    checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=TumorNet(checkpoint['variant'],checkpoint['width'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    with h5py.File(source) as f:
        image=f['cjdata/image'][()].T.astype(np.float32)
    hi=float(np.percentile(image[image>0],99)) if np.any(image>0) else 1
    normalized=np.clip(image/max(hi,1),0,1)
    size=checkpoint['config'].get('input_size',128)
    x=F.interpolate(torch.from_numpy(normalized)[None,None],size=(size,size),mode='bilinear',align_corners=False)
    out=model(x)
    probability=F.interpolate(out['seg_logits'].sigmoid(),size=image.shape,mode='bilinear',align_corners=False)[0,0].numpy()
    mask=probability>=.5
    yy,xx=np.where(mask)
    box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
    classes=['meningioma','glioma','pituitary']
    scores=out['class_logits'].softmax(1)[0].tolist()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rgb=np.repeat((normalized*255).astype(np.uint8)[:,:,None],3,axis=2)
    overlay=rgb.copy();overlay[mask]=(overlay[mask]*.55+np.array([255,75,90])*.45).astype(np.uint8)
    Image.fromarray(overlay).save(output/'overlay.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(output/'mask.png')
    np.save(output/'probability.npy',probability)
    result={'input':str(source),'variant':checkpoint['variant'],'seed':checkpoint['seed'],'predicted_class':classes[int(np.argmax(scores))],'class_probabilities':dict(zip(classes,scores)),'mask_derived_bbox_xyxy_exclusive':box,'notes':f'Research demo trained on tumor-positive slices only. Bounding box comes from the segmentation mask, not an object detector. Full-resolution display uses interpolated {size}x{size} predictions.'}
    (output/'prediction.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--output',default='output/demo');a=p.parse_args();predict(a.checkpoint,a.input,a.output)
