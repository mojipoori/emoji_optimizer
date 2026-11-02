#mrpc
python -m src.main --task mrpc --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 1402
python -m src.main --task mrpc --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 1993
python -m src.main --task mrpc --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 2025
python -m src.main --task mrpc --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 8
python -m src.main --task mrpc --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979

python -m src.main --task mrpc --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 1402
python -m src.main --task mrpc --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 1993
python -m src.main --task mrpc --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 2025
python -m src.main --task mrpc --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 8
python -m src.main --task mrpc --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979

python -m src.main --task mrpc --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 1402 
python -m src.main --task mrpc --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 1993 
python -m src.main --task mrpc --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 2025 
python -m src.main --task mrpc --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 8 
python -m src.main --task mrpc --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979 

python -m src.main --task mrpc --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 1402 --lion_weight_decay 0.1
python -m src.main --task mrpc --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 1993 --lion_weight_decay 0.1
python -m src.main --task mrpc --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 2025 --lion_weight_decay 0.1
python -m src.main --task mrpc --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 8 --lion_weight_decay 0.1
python -m src.main --task mrpc --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 11021979 --lion_weight_decay 0.1



#cola
python -m src.main --task cola --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 1402
python -m src.main --task cola --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 1993
python -m src.main --task cola --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 2025
python -m src.main --task cola --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 8
python -m src.main --task cola --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979

python -m src.main --task cola --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 1402
python -m src.main --task cola --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 1993
python -m src.main --task cola --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 2025
python -m src.main --task cola --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 8
python -m src.main --task cola --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979

python -m src.main --task cola --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 1402 
python -m src.main --task cola --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 1993 
python -m src.main --task cola --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 2025 
python -m src.main --task cola --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 8 
python -m src.main --task cola --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979 

python -m src.main --task cola --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 1e-6 --seed 1402 --lion_weight_decay 0.1
python -m src.main --task cola --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 1e-6 --seed 1993 --lion_weight_decay 0.1
python -m src.main --task cola --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 1e-6 --seed 2025 --lion_weight_decay 0.1
python -m src.main --task cola --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 1e-6 --seed 8 --lion_weight_decay 0.1
python -m src.main --task cola --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 1e-6 --seed 11021979 --lion_weight_decay 0.1



#sts-b
python -m src.main --task sts --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 1402
python -m src.main --task sts --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 1993
python -m src.main --task sts --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 2025
python -m src.main --task sts --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 8
python -m src.main --task sts --optimizer adabelief --adabelief_betas 0.9 0.999 --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979

python -m src.main --task sts --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 1402
python -m src.main --task sts --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 1993
python -m src.main --task sts --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 2025
python -m src.main --task sts --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 8
python -m src.main --task sts --optimizer adamw --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979

python -m src.main --task sts --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 1402 
python -m src.main --task sts --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 1993 
python -m src.main --task sts --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 2025 
python -m src.main --task sts --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 8 
python -m src.main --task sts --optimizer emoji --batch_size 8 --gradient_accumulation_steps 1 --seed 11021979 

python -m src.main --task sts --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 1402 --lion_weight_decay 0.1
python -m src.main --task sts --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 1993 --lion_weight_decay 0.1
python -m src.main --task sts --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 2025 --lion_weight_decay 0.1
python -m src.main --task sts --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 8 --lion_weight_decay 0.1
python -m src.main --task sts --optimizer lion --lion_betas 0.9 0.99 --batch_size 8 --gradient_accumulation_steps 1 --lion_lr 2e-6 --seed 11021979 --lion_weight_decay 0.1
