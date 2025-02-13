# Reproduce

### shoes2edges

### train
python train.py --name shoes2edges --target_dataroot ./datasets/edges2shoes --source_dataroot ./datasets/edges2handbags --target_max_dataset_size 40000 --source_max_dataset_size 120000 --direction BtoA --model pix2pix --gan_mode wgangp --display_id -1 --batch_size 16 --OT_weight 0.1 --OT_weight1 1.0 --OT_weight2 1.0
python train.py --name shoes2edges-TO --target_dataroot ./datasets/edges2shoes --source_dataroot ./datasets/edges2handbags --target_max_dataset_size 40000 --source_max_dataset_size 120000 --direction BtoA --model pix2pix --gan_mode wgangp --display_id -1 --batch_size 16 --OT_weight 0.1 --OT_weight1 1.0 --OT_weight2 1.0 --target_only

### test
python test.py --target_dataroot ./datasets/edges2shoes --source_dataroot ./datasets/edges2handbags --direction AtoB --model pix2pix --name shoes2edges --phase val --serial_batches
python test.py --target_dataroot ./datasets/edges2shoes --source_dataroot ./datasets/edges2handbags --direction AtoB --model pix2pix --name shoes2edges-TO --phase val --serial_batches

----

### edges2shoes

### train
python train.py --name edges2shoes --target_dataroot ./datasets/edges2shoes --source_dataroot ./datasets/edges2handbags --target_max_dataset_size 40000 --source_max_dataset_size 120000 --direction AtoB --model pix2pix --gan_mode lsgan --display_id -1 --batch_size 16 --OT_weight 0.1 --OT_weight1 1.0 --OT_weight2 1.0
python train.py --name edges2shoes-TO --target_dataroot ./datasets/edges2shoes --source_dataroot ./datasets/edges2handbags --target_max_dataset_size 40000 --source_max_dataset_size 120000 --direction AtoB --model pix2pix --gan_mode lsgan --display_id -1 --batch_size 16 --OT_weight 0.1 --OT_weight1 1.0 --OT_weight2 1.0 --target_only

### test
python test.py --target_dataroot ./datasets/edges2shoes --source_dataroot ./datasets/edges2handbags --direction AtoB --model pix2pix --name edges2shoes --phase val --serial_batches
python test.py --target_dataroot ./datasets/edges2shoes --source_dataroot ./datasets/edges2handbags --direction AtoB --model pix2pix --name edges2shoes-TO --phase val --serial_batches

---

### Reference

Original Code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix