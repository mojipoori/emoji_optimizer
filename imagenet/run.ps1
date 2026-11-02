python main_when_copy_2.py --optimizer adabelief --lr 1e-3 --wd 1e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decouple True --batch-size 256 --arch resnet18 --when 30 60 90 --seed 42

python main_when_copy_2.py --optimizer emoji --lr 1e-3 --wd 1e-2 --emoji_betas 0.9 0.999 --emoji_eps 1e-8 --emoji_weight_decouple True --batch-size 256 --arch resnet18 --when 30 60 90 --seed 42
