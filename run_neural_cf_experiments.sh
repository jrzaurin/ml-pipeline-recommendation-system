python gmf.py --lr_scheduler --save_results
python gmf.py --n_emb 16 --lr_scheduler --save_results
python gmf.py --n_emb 32 --lr_scheduler --save_results

python mlp.py --lr_scheduler --save_results
python mlp.py --layers "[64, 32, 16]"  --lr_scheduler --save_results
python mlp.py --layers "[64, 32, 16]" --dropouts "[0.5, 0.5]" --lr_scheduler --save_results

python neumf.py --n_emb 16 --layers "[64, 32, 16]" --dropouts "[0.,0.]" \
--mf_pretrain "gmf_2020-11-20_10:09:11.747207.pt" --mlp_pretrain "mlp_2020-11-21_10:47:34.310745.pt" \
--n_epochs 20 --lr_scheduler --save_results

python neumf.py --n_emb 16 --layers "[64, 32, 16]" --dropouts "[0.,0.]" --lr 0.01 \
--mf_pretrain "gmf_2020-11-20_10:09:11.747207.pt" --mlp_pretrain "mlp_2020-11-21_10:47:34.310745.pt" \
--n_epochs 5 --lr_scheduler --freeze --eval_every 1 --save_results

python gmf.py --lr_scheduler --concat --save_results
python gmf.py --n_emb 16 --lr_scheduler --concat --save_results
python gmf.py --n_emb 32 --lr_scheduler --concat --save_results
