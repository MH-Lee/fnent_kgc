
# mode, model, dataset, GPU_ID, SAVE_ID, BSz, Neg_sample, Hidden_dim, g, a, lr, epoch, test_bsz
nohup bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 -de > ./output/rotate_FB15k.out &
nohup bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de > ./output/rotate_FB15k237.out &
nohup bash run.sh train RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de > ./output/rotate_WN18.out &
nohup bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de > ./output/rotate_WN18RR.out &
nohup bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de > ./output/rotate_YAGO3-10.out &
nohup bash run.sh train RotatE YAGO3-10-DR 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de > ./output/rotate_YAGO3-10-DR.out &	
nohup bash run.sh train TransE YAGO3-10 3 0 1024 400 500 24.0 1.0 0.0002 100000 4 > ./output/transe_YAGO3-10.out &
nohup bash run.sh train ComplEx YAGO3-10 3 0 1024 400 500 500.0 1.0 0.002 100000 4 -de -dr -r 0.000002  > ./output/complex_YAGO3-10.out &
nohup bash run.sh train DistMult YAGO3-10 3 0 1024 400 500 500.0 1.0 0.002 100000 4 -r 0.000002 > ./output/distmult_YAGO3-10.out &
nohup bash run.sh train TransE YAGO3-10-DR 3 0 1024 400 500 24.0 1.0 0.0002 100000 4 > ./output/transe_YAGO3-10-DR.out &
nohup bash run.sh train ComplEx YAGO3-10-DR 3 0 1024 400 500 500.0 1.0 0.002 100000 4 -de -dr -r 0.000002  > ./output/complex_YAGO3-10-DR.out &
nohup bash run.sh train DistMult YAGO3-10-DR 3 0 1024 400 500 500.0 1.0 0.002 100000 4 -r 0.000002 > ./output/distmult_YAGO3-10-DR.out &