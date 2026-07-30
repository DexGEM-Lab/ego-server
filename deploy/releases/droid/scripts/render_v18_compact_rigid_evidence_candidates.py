#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-report', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--top-k', type=int, default=24)
    ap.add_argument('--panel-width', type=int, default=240)
    ap.add_argument('--panel-height', type=int, default=160)
    args=ap.parse_args()
    r=load_json(args.evidence_report)
    cands=r['all_candidate_frames'][:args.top_k]
    panels=[]
    for c in cands:
        raw=Image.open(c['raw_frame_path']).convert('RGB')
        mask=Image.open(c['mask_path']).convert('L').resize(raw.size, Image.Resampling.NEAREST)
        panel=raw.copy().convert('RGBA')
        overlay=Image.new('RGBA', raw.size, (0,0,0,0))
        orange=Image.new('RGBA', raw.size, (255,140,0,70))
        overlay=Image.composite(orange, overlay, mask)
        panel=Image.alpha_composite(panel, overlay).convert('RGB')
        arr=np.asarray(mask)>0
        d=ImageDraw.Draw(panel)
        if arr.any():
            ys,xs=np.where(arr)
            d.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255,200,0), width=3)
        d.rectangle([0,0,raw.width,44], fill=(0,0,0))
        d.text((5,4), f"f{c['frame_idx']} depth {c['visible_depth_vertex_count']} mask {c['mask_area_px']}", fill=(255,255,255))
        panel=panel.resize((args.panel_width,args.panel_height), Image.Resampling.BILINEAR)
        panels.append(panel)
    cols=min(4,len(panels)); rows=int(np.ceil(len(panels)/cols))
    sheet=Image.new('RGB',(cols*args.panel_width, rows*args.panel_height),(20,20,20))
    for i,p in enumerate(panels): sheet.paste(p,((i%cols)*args.panel_width,(i//cols)*args.panel_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    summary={'method':'render_v18_compact_rigid_evidence_candidates','status':'ok','output':str(args.output),'frames':[c['frame_idx'] for c in cands]}
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__=='__main__': main()
