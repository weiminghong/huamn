import json, math, os, statistics, random
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

ANN='/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json'
IMGROOT='/root/autodl-tmp/coco/val2017'
PRED='/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_flatpose/full64115_3ep_bs8ga2_alllinear_20260723/generated_val_full_epoch1_bs2/predictions.coco_multitask.json'
OUTDIR='/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_flatpose/full64115_3ep_bs8ga2_alllinear_20260723/generated_val_full_epoch1_bs2/diagnostics_pose_seg'
os.makedirs(OUTDIR, exist_ok=True)

def xywh_to_xyxy(b):
    return [float(b[0]), float(b[1]), float(b[0])+float(b[2]), float(b[1])+float(b[3])]

def iou_xyxy(a,b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-9)

def greedy_match(preds, gts, key_fn, thr=0.0):
    pairs=[]
    for pi,p in enumerate(preds):
        for gi,g in enumerate(gts):
            v=key_fn(p,g)
            if v>=thr: pairs.append((v,pi,gi))
    pairs.sort(reverse=True)
    usedp=set(); usedg=set(); out=[]
    for v,pi,gi in pairs:
        if pi in usedp or gi in usedg: continue
        usedp.add(pi); usedg.add(gi); out.append((pi,gi,v))
    return out

def gt_keypoints(ann):
    kp=ann.get('keypoints') or ann.get('keypoints_2d') or []
    if len(kp)==17 and isinstance(kp[0], (list,tuple)):
        flat=[]
        for item in kp:
            flat.extend([float(item[0]), float(item[1]), float(item[2] if len(item)>2 else 2)])
        return flat
    return [float(x) for x in kp] if len(kp)>=51 else []

def pck(pred_kp, gt_kp, norm_box, alpha=0.2):
    if not pred_kp or not gt_kp or len(pred_kp)<51 or len(gt_kp)<51: return (0,0)
    w=max(1.0, norm_box[2]-norm_box[0]); h=max(1.0, norm_box[3]-norm_box[1]); norm=max(w,h)
    ok=0; total=0
    for i in range(17):
        gx,gy,gv=gt_kp[3*i], gt_kp[3*i+1], gt_kp[3*i+2]
        if gv <= 0: continue
        px,py=pred_kp[3*i], pred_kp[3*i+1]
        total += 1
        if math.hypot(px-gx, py-gy) <= alpha*norm: ok += 1
    return ok,total

def kp_inside_box(kp, box):
    if not kp: return (0,0)
    inside=0; total=0
    x1,y1,x2,y2=box
    for i in range(17):
        x,y,v=kp[3*i],kp[3*i+1],kp[3*i+2]
        if v <= 0: continue
        total += 1
        if x1 <= x <= x2 and y1 <= y <= y2: inside += 1
    return inside,total

def seg_bbox(seg):
    if not seg: return None
    try:
        rle=seg
        if isinstance(rle, list):
            return None
        bb=mask_utils.toBbox(rle).tolist()
        return xywh_to_xyxy(bb)
    except Exception:
        return None

ann=json.load(open(ANN))
imgs={im['id']: im for im in ann['images']}
gt_by_img=defaultdict(list)
for a in ann['annotations']:
    if a.get('iscrowd',0): continue
    valid=a.get('valid') or {}
    if 'bboxes' in valid and not int(valid.get('bboxes',1)): continue
    a['_bbox_xyxy']=xywh_to_xyxy(a['bbox'])
    a['_keypoints']=gt_keypoints(a)
    gt_by_img[a['image_id']].append(a)

preds=json.load(open(PRED))
pred_by_img=defaultdict(list)
for p in preds:
    p['_bbox_xyxy']=xywh_to_xyxy(p['bbox'])
    p['_seg_bbox_xyxy']=seg_bbox(p.get('segmentation'))
    pred_by_img[p['image_id']].append(p)

stats={
 'images': len(imgs), 'preds': len(preds), 'gt': sum(len(v) for v in gt_by_img.values()),
 'bbox_match_iou50':0, 'bbox_match_all':0, 'bbox_pck_ok':0, 'bbox_pck_total':0,
 'bbox_match_iou75':0, 'bbox75_pck_ok':0, 'bbox75_pck_total':0,
 'kp_inside_pred_ok':0, 'kp_inside_pred_total':0,
 'kp_inside_gt_ok':0, 'kp_inside_gt_total':0,
 'seg_bbox_iou_values':[], 'seg_bbox_iou_low':0,
 'pred_box_area':[], 'seg_box_area':[],
 'examples_bad':[]
}
per_img=[]
for image_id,gts in gt_by_img.items():
    ps=pred_by_img.get(image_id, [])
    matches=greedy_match(ps,gts,lambda p,g: iou_xyxy(p['_bbox_xyxy'],g['_bbox_xyxy']),thr=0.0)
    img_ok=img_tot=0
    for pi,gi,ov in matches:
        p=ps[pi]; g=gts[gi]
        if ov >= 0.5:
            stats['bbox_match_iou50'] += 1
            ok,tot=pck(p.get('keypoints'),g.get('_keypoints'),g['_bbox_xyxy'])
            stats['bbox_pck_ok'] += ok; stats['bbox_pck_total'] += tot
            img_ok += ok; img_tot += tot
        if ov >= 0.75:
            stats['bbox_match_iou75'] += 1
            ok,tot=pck(p.get('keypoints'),g.get('_keypoints'),g['_bbox_xyxy'])
            stats['bbox75_pck_ok'] += ok; stats['bbox75_pck_total'] += tot
    stats['bbox_match_all'] += len(matches)
    for p in ps:
        kp=p.get('keypoints') or []
        ok,tot=kp_inside_box(kp,p['_bbox_xyxy'])
        stats['kp_inside_pred_ok'] += ok; stats['kp_inside_pred_total'] += tot
        sb=p.get('_seg_bbox_xyxy')
        if sb:
            ov=iou_xyxy(p['_bbox_xyxy'],sb)
            stats['seg_bbox_iou_values'].append(ov)
            if ov < 0.3: stats['seg_bbox_iou_low'] += 1
            stats['pred_box_area'].append(max(0,p['_bbox_xyxy'][2]-p['_bbox_xyxy'][0])*max(0,p['_bbox_xyxy'][3]-p['_bbox_xyxy'][1]))
            stats['seg_box_area'].append(max(0,sb[2]-sb[0])*max(0,sb[3]-sb[1]))
    for pi,gi,ov in matches[:len(gts)]:
        p=ps[pi]; g=gts[gi]
        ok,tot=kp_inside_box(p.get('keypoints') or [], g['_bbox_xyxy'])
        stats['kp_inside_gt_ok'] += ok; stats['kp_inside_gt_total'] += tot
    if img_tot:
        per_img.append((img_ok/img_tot, image_id, len(ps), len(gts)))

summary={}
for k,v in stats.items():
    if isinstance(v, list): continue
    summary[k]=v
summary.update({
 'bbox_matched_pck@0.2': stats['bbox_pck_ok']/max(1,stats['bbox_pck_total']),
 'bbox75_matched_pck@0.2': stats['bbox75_pck_ok']/max(1,stats['bbox75_pck_total']),
 'kp_visible_inside_pred_bbox_rate': stats['kp_inside_pred_ok']/max(1,stats['kp_inside_pred_total']),
 'kp_visible_inside_bbox_matched_gt_bbox_rate': stats['kp_inside_gt_ok']/max(1,stats['kp_inside_gt_total']),
 'seg_bbox_vs_pred_bbox_iou_mean': float(np.mean(stats['seg_bbox_iou_values'])) if stats['seg_bbox_iou_values'] else None,
 'seg_bbox_vs_pred_bbox_iou_median': float(np.median(stats['seg_bbox_iou_values'])) if stats['seg_bbox_iou_values'] else None,
 'seg_bbox_vs_pred_bbox_iou_lt_0.3_rate': stats['seg_bbox_iou_low']/max(1,len(stats['seg_bbox_iou_values'])),
 'pred_box_area_mean': float(np.mean(stats['pred_box_area'])) if stats['pred_box_area'] else None,
 'seg_box_area_mean': float(np.mean(stats['seg_box_area'])) if stats['seg_box_area'] else None,
})
# select worst pose visible examples among bbox matched images
per_img.sort()
summary['worst_pose_images_by_bboxmatch_pck']=[{'image_id':i,'pck':p,'preds':pp,'gts':gg} for p,i,pp,gg in per_img[:20]]
json.dump(summary, open(os.path.join(OUTDIR,'summary.json'),'w'), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])

# visualizations: worst + duplicate/error health images if present in inference.jsonl
image_ids=[x['image_id'] for x in summary['worst_pose_images_by_bboxmatch_pck'][:12]]
# add known bad ids if available
for i in [326541,558114,399205,427256,411754,233238]:
    if i in imgs and i not in image_ids: image_ids.append(i)
image_ids=image_ids[:18]
for image_id in image_ids:
    im=imgs[image_id]
    path=os.path.join(IMGROOT, im.get('file_name', f'{image_id:012d}.jpg'))
    if not os.path.exists(path):
        path=os.path.join(IMGROOT, f'{image_id:012d}.jpg')
    try:
        img=Image.open(path).convert('RGB')
    except Exception:
        continue
    draw=ImageDraw.Draw(img, 'RGBA')
    # GT green, pred red, seg bbox blue, pose yellow
    for g in gt_by_img.get(image_id,[]):
        x1,y1,x2,y2=g['_bbox_xyxy']; draw.rectangle([x1,y1,x2,y2], outline=(0,255,0,220), width=3)
        kp=g.get('_keypoints') or []
        for j in range(17):
            if len(kp)>=51 and kp[3*j+2]>0:
                x,y=kp[3*j],kp[3*j+1]; draw.ellipse([x-2,y-2,x+2,y+2], fill=(0,255,0,220))
    for p in pred_by_img.get(image_id,[])[:25]:
        x1,y1,x2,y2=p['_bbox_xyxy']; draw.rectangle([x1,y1,x2,y2], outline=(255,0,0,220), width=2)
        sb=p.get('_seg_bbox_xyxy')
        if sb: draw.rectangle(sb, outline=(0,128,255,180), width=1)
        kp=p.get('keypoints') or []
        for j in range(17):
            if len(kp)>=51 and kp[3*j+2]>0:
                x,y=kp[3*j],kp[3*j+1]; draw.ellipse([x-2,y-2,x+2,y+2], fill=(255,255,0,230))
    img.save(os.path.join(OUTDIR, f'{image_id}.jpg'), quality=90)
print('saved_dir', OUTDIR)
