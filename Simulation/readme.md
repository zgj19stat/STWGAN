# Reproduce

### for example: seed 42

### Ours
python main.py --result ./M1 --seed 42 --warmup 300 --epochs 600 --datasets M1 --is-test 
python main.py --result ./M2 --seed 42 --warmup 300 --epochs 350 --datasets M2 --is-test 
python main.py --result ./M3 --seed 42 --warmup 300 --epochs 400 --datasets M3 --is-test 

### Target-Only
python main.py --result ./M1-TO --seed 42 --warmup 300 --epochs 600 --datasets M1 --is-test --target-only 
python main.py --result ./M2-TO --seed 42 --warmup 300 --epochs 350 --datasets M2 --is-test --target-only 
python main.py --result ./M3-TO --seed 42 --warmup 300 --epochs 400 --datasets M3 --is-test --target-only 

### Pool
python main.py --result ./M1-Pool --seed 42 --warmup 300 --epochs 600 --datasets M1 --is-test --OT-weight 0.
python main.py --result ./M2-Pool --seed 42 --warmup 300 --epochs 350 --datasets M2 --is-test --OT-weight 0.
python main.py --result ./M3-Pool --seed 42 --warmup 300 --epochs 400 --datasets M3 --is-test --OT-weight 0.

### Pretrained
python main.py --result ./M1-PTFT --seed 42 --warmup 450 --epochs 600 --datasets M1 --is-test --pretrained
python main.py --result ./M2-PTFT --seed 42 --warmup 450 --epochs 600 --datasets M2 --is-test --pretrained
python main.py --result ./M3-PTFT --seed 42 --warmup 450 --epochs 600 --datasets M3 --is-test --pretrained
