python gmf.py --lr_scheduler --save_results
python gmf.py --n_emb 16 --lr_scheduler --save_results
python gmf.py --n_emb 32 --lr_scheduler --save_results

python mlp.py --lr_scheduler --save_results
python mlp.py --layers "[64, 32, 16]"  --lr_scheduler --save_results
python mlp.py --layers "[64, 32, 16]" --dropouts "[0.5, 0.5]" --lr_scheduler --save_results

# python neumf.py --batch_size 1024 --lr 0.01 --n_emb 8 --lr_scheduler --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
# --mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
# --mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
# --epochs 20 --learner "SGD" --validate_every 2

# python neumf.py --batch_size 1024 --lr 0.001 --n_emb 8 --layers "[128, 64, 32]" --dropouts "[0.,0.]" \
# --mf_pretrain "GMF_bs_1024_lr_001_n_emb_8_lrnr_adam_lrs_wolrs.pt" \
# --mlp_pretrain "MLP_bs_1024_reg_00_lr_003_n_emb_64_ll_32_dp_wodp_lrnr_adam_lrs_wlrs.pt" \
# --freeze 1 --epochs 4

