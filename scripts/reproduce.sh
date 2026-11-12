#!/bin/bash
# Reproduce the README results. Run from the repository root, one step at a time:
#   bash scripts/reproduce.sh data        # annotation jsons + rotated COCO val (after organizing the datasets)
#   bash scripts/reproduce.sh pretrain    # COCO pretraining
#   bash scripts/reproduce.sh coco        # COCO-pretrained table (ARPD + RAPiD COCO608)
#   bash scripts/reproduce.sh finetune    # three leave-one-dataset-out fine-tunes
#   bash scripts/reproduce.sh fisheye     # fisheye table (ARPD + RAPiD HBCP608)
#   bash scripts/reproduce.sh speed       # speed table
#   bash scripts/reproduce.sh table       # print the tables from the metrics files
# Environment: RAPID (RAPiD repo, default ../RAPiD), NGPU (pretraining processes, default 2).
set -euo pipefail
RAPID=${RAPID:-../RAPiD}
NGPU=${NGPU:-2}
C=data/coco
P=runs/coco_pretrain_dla34_512/final.pth
COCO_VAL="coco_val:$C/val2017:$C/annotations/person_val2017_rot.json"
COCO_ROT="coco_val_rot180:$C/val2017_rot180_s0:$C/annotations/person_val2017_rot180_s0.json"

json_test() {  # name img_dir ann -> data.test override for tools/evaluate.py
  echo "data.test=[{type: json, name: $1, img_dir: $2, ann: $3}]"
}

case ${1:-} in
data)
  python tools/prepare_data.py coco_person --coco $C
  python tools/prepare_data.py monitor --coco $C --root data/fisheye
  python tools/make_rotated_val.py --ann $C/annotations/person_val2017_rot.json --img_dir $C/val2017 \
    --out_img $C/val2017_rot180_s0 --out_ann $C/annotations/person_val2017_rot180_s0.json
  ;;
pretrain)
  torchrun --standalone --nproc_per_node=$NGPU tools/train.py configs/coco_pretrain.yaml
  ;;
coco)
  python tools/evaluate.py $P --tag coco_val "$(json_test coco_val $C/val2017 $C/annotations/person_val2017_rot.json)"
  python tools/evaluate.py $P --tag coco_val_rot180 \
    "$(json_test coco_val_rot180 $C/val2017_rot180_s0 $C/annotations/person_val2017_rot180_s0.json)"
  for D in HABBOF CEPDOF WEPDTOF; do
    python tools/evaluate.py $P --tag zeroshot_$D "data.test=[{type: fisheye, dataset: $D}]"
  done
  python tools/eval_rapid.py $RAPID/weights/rapid_pL1_dark53_COCO608_Jan07_20000.ckpt --rapid $RAPID \
    --tag COCO608 --json $COCO_VAL $COCO_ROT --datasets HABBOF CEPDOF WEPDTOF
  ;;
finetune)
  for T in HABBOF CEPDOF WEPDTOF; do
    python tools/train.py configs/cv_test_$T.yaml
  done
  ;;
fisheye)
  for T in HABBOF CEPDOF WEPDTOF; do
    python tools/evaluate.py runs/cv_test_$T/final.pth --tag in1024 input_size=1024
  done
  python tools/evaluate.py runs/cv_test_WEPDTOF/epoch1.pth --tag ep1
  for S in 608 1024; do
    python tools/eval_rapid.py $RAPID/weights/pL1_HBCP608_Apr14_6000.ckpt --rapid $RAPID \
      --datasets WEPDTOF --input_size $S --tag HBCP608_in$S
  done
  ;;
speed)
  python tools/benchmark.py --ckpt runs/cv_test_WEPDTOF/final.pth --sizes 512 608 1024 --dtype fp32 \
    --image data/fisheye/WEPDTOF/exhibition/exhibition_000001.jpg \
    --rapid $RAPID --rapid_ckpt $RAPID/weights/pL1_HBCP608_Apr14_6000.ckpt
  ;;
table)
  python - <<'EOF'
import json, os
def row(name, path, pr=False):
    if not os.path.exists(path):
        print(f'{name:40s} (missing {path})'); return
    m = json.load(open(path))['mean']
    extra = f" {m['P']:.3f} {m['R']:.3f}" if pr else ''
    print(f"{name:40s} {100*m['AP50']:5.1f} {100*m['AP']:5.1f} {100*m['AP75']:5.1f}{extra}")
A, R = 'runs/coco_pretrain_dla34_512', 'runs/rapid_baseline'
print('COCO-pretrained                          AP50    AP  AP75     P     R')
for s, a in [('coco_val', 'coco_val'), ('coco_val_rot180', 'coco_val_rot180'),
             ('HABBOF', 'zeroshot_HABBOF'), ('CEPDOF', 'zeroshot_CEPDOF'), ('WEPDTOF', 'zeroshot_WEPDTOF')]:
    row(f'ARPD  {s}', f'{A}/metrics_{a}.json', True)
    row(f'RAPiD {s}', f'{R}/metrics_COCO608_{s}.json', True)
print('\nFisheye                                  AP50    AP  AP75')
for T in ('HABBOF', 'CEPDOF'):
    row(f'ARPD  {T} 512 (10 ep)', f'runs/cv_test_{T}/metrics_final.json')
    row(f'ARPD  {T} 1024 (10 ep)', f'runs/cv_test_{T}/metrics_in1024.json')
row('RAPiD WEPDTOF 608', f'{R}/metrics_HBCP608_in608_WEPDTOF.json')
row('ARPD  WEPDTOF 512 (1 ep)', 'runs/cv_test_WEPDTOF/metrics_ep1.json')
row('RAPiD WEPDTOF 1024', f'{R}/metrics_HBCP608_in1024_WEPDTOF.json')
row('ARPD  WEPDTOF 1024 (10 ep)', 'runs/cv_test_WEPDTOF/metrics_in1024.json')
EOF
  ;;
*)
  sed -n 2,10p "$0"; exit 1
  ;;
esac
