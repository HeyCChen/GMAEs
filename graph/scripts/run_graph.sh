for dataset in IMDB-BINARY
do
	python -W ignore main_graph.py \
		--device 0 \
		--dataset $dataset \
		--pe_dim 32 \
		--mask_rate 0.25 \
		--encoder "gin" \
		--decoder "gin" \
		--in_drop 0.2 \
		--attn_drop 0.1 \
		--num_layers 2 \
		--num_hidden 512 \
		--num_heads 2 \
		--max_epoch 100 \
		--max_epoch_f 0 \
		--lr 0.00015 \
		--weight_decay 0.0 \
		--activation prelu \
		--optimizer adam \
		--drop_edge_rate 0.0 \
		--loss_fn "sce" \
		--seeds 0 1 2 3 4 \
		--linear_prob \
		--num_subgraph 0 \
		--use_cfg \

done


