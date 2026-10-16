#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_coco(path):
    with open(path, 'r') as f:
        return json.load(f)


def clamp_box_xyxy(box, width, height):
    x, y, w, h = box
    x1 = max(0.0, min(float(width), float(x)))
    y1 = max(0.0, min(float(height), float(y)))
    x2 = max(0.0, min(float(width), float(x) + float(w)))
    y2 = max(0.0, min(float(height), float(y) + float(h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def convert(annotation, image_root, output, limit=None, min_area=1.0):
    data = load_coco(annotation)
    images = {im['id']: im for im in data.get('images', [])}
    anns_by_image = defaultdict(list)
    total_valid_boxes = 0
    skipped_invalid = 0
    for ann in data.get('annotations', []):
        if ann.get('iscrowd', 0):
            skipped_invalid += 1
            continue
        valid = ann.get('valid') or {}
        if valid and not valid.get('bboxes', 1):
            skipped_invalid += 1
            continue
        im = images.get(ann.get('image_id'))
        if im is None:
            skipped_invalid += 1
            continue
        box = clamp_box_xyxy(ann.get('bbox', [0, 0, 0, 0]), im['width'], im['height'])
        if box is None:
            skipped_invalid += 1
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) < min_area:
            skipped_invalid += 1
            continue
        anns_by_image[ann['image_id']].append((ann.get('id', 0), box))
        total_valid_boxes += 1

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count_images = 0
    count_boxes = 0
    missing_images = 0
    with output.open('w') as f:
        for image_id in sorted(images):
            im = images[image_id]
            boxes = [box for _, box in sorted(anns_by_image.get(image_id, []), key=lambda x: x[0])]
            if not boxes:
                continue
            image_path = Path(image_root) / im['file_name']
            if not image_path.exists():
                missing_images += 1
                continue
            refs = ['person'] * len(boxes)
            # Keep the assistant target exactly in ms-swift grounding style. The actual coords live in objects.bbox.
            answer = '[' + ', '.join('{"bbox_2d": <bbox>, "label": "<ref-object>"}' for _ in boxes) + ']'
            row = {
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': '<image>Locate every person in the image.'},
                    {'role': 'assistant', 'content': answer},
                ],
                'images': [str(image_path)],
                'objects': {'ref': refs, 'bbox': boxes, 'bbox_type': 'real'},
                'meta': {'image_id': image_id, 'width': im['width'], 'height': im['height'], 'num_persons': len(boxes)},
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count_images += 1
            count_boxes += len(boxes)
            if limit and count_images >= limit:
                break
    return {
        'annotation': str(annotation),
        'output': str(output),
        'images_in': len(images),
        'annotations_in': len(data.get('annotations', [])),
        'images_out': count_images,
        'boxes_out': count_boxes,
        'total_valid_boxes_before_limit': total_valid_boxes,
        'skipped_invalid_annotations': skipped_invalid,
        'missing_images': missing_images,
        'limit': limit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotation', required=True)
    ap.add_argument('--image-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    stats = convert(args.annotation, args.image_root, args.output, args.limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
