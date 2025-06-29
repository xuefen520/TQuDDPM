#python TQuDDPM_main.py.py --na 1 --Encode_type "Ry-2pi" --L 8
#python TQuDDPM_main.py.py --na 1 --Encode_type "Ry-pi" --L 8
#python TQuDDPM_main.py.py --na 1 --Encode_type "Rx-2pi" --L 8
#python TQuDDPM_main.py.py --na 1 --Encode_type "Rx-pi" --L 8
#
#python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-2pi" --L 8



#python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-pi" --L 8
#python TQuDDPM_main.py.py --na 2 --Encode_type "Rx-2pi" --L 8
#python TQuDDPM_main.py.py --na 2 --Encode_type "Rx-pi" --L 8
#
#python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-2pi" --L 6
#python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-pi" --L 6
#python TQuDDPM_main.py.py --na 2 --Encode_type "Rx-2pi" --L 6
#python TQuDDPM_main.py.py --na 2 --Encode_type "Rx-pi" --L 6


for s in 7
do
for l in 4 8 12 16 20
do
  python Generation_data_circle.py --n 1 --na 1 --T 20 --Ndata 100 --Encode_type "Ry-pi-Rz-2pi" --batch_size 4 --L $l --seed $s  --lr 0.0005 --weight_t0 0.01  --problem "cluster" --loss_type "MMD" --epochs 20000
done
done



#for s in 1
#do
#for l in 22
#do
#  python TQuDDPM_main.py.py --n 4 --na 2 --T 30 --Ndata 100 --Encode_type "NN" --batch_size 4 --L $l --seed $s  --lr 0.05 --weight_t0 1.0 --problem "phase" --loss_type "MMD" --epochs 20000
#done
#done

#python TQuDDPM_main.py.py --n 4 --na 2 --T 30 --Ndata 100 --Encode_type "NN" --batch_size 4 --L 22  --lr 0.005 --weight_t0 0.01 --problem "phase" --loss_type "MMD" --epochs 20000  --seed 1
#for s in 2
#do
#for wt in 1.0 0.5 0.1 0.05 0.01
#do
#  python TQuDDPM_main.py.py --n 4 --na 2 --T 30 --Ndata 100 --Encode_type "NN" --batch_size 4 --L 16 --seed $s  --lr 0.0005 --weight_t0 $wt --problem "phase" --loss_type "MMD" --epochs 10000
#done
#done
#
#for s in 3
#do
#for wt in 1.0 0.5 0.1 0.05 0.01
#do
#  python TQuDDPM_main.py.py --n 4 --na 2 --T 30 --Ndata 100 --Encode_type "NN" --batch_size 4 --L 16 --seed $s  --lr 0.0005 --weight_t0 $wt --problem "phase" --loss_type "MMD" --epochs 10000
#done
#done
#
#for s in 4
#do
#for wt in 1.0 0.5 0.1 0.05 0.01
#do
#  python TQuDDPM_main.py.py --n 4 --na 2 --T 30 --Ndata 100 --Encode_type "NN" --batch_size 4 --L 16 --seed $s  --lr 0.0005 --weight_t0 $wt --problem "phase" --loss_type "MMD" --epochs 10000
#done
#done
#
#for s in 5
#do
#for wt in 1.0 0.5 0.1 0.05 0.01
#do
#  python TQuDDPM_main.py.py --n 4 --na 2 --T 30 --Ndata 100 --Encode_type "NN" --batch_size 4 --L 16 --seed $s  --lr 0.0005 --weight_t0 $wt --problem "phase" --loss_type "MMD" --epochs 10000
#done
#done

#for s in 3 4
#do
#for l in 6 10 14 18
#do
#  python TQuDDPM_main.py.py --n 2 --na 1 --T 20 --Ndata 100 --Encode_type "Ry-pi" --batch_size 4 --L $l --seed $s  --lr 0.0005 --weight_t0 0.01 --problem "cluster" --loss_type "MMD"
#done
#done
#
#for s in 5
#do
#for l in 6 10 14 18
#do
#  python TQuDDPM_main.py.py --n 2 --na 1 --T 20 --Ndata 100 --Encode_type "Ry-pi" --batch_size 4 --L $l --seed $s  --lr 0.0005 --weight_t0 0.01 --problem "cluster" --loss_type "MMD"
#done
#done
#
#for s in 1
#do
#for l in 6 10 14 18
#do
#  python TQuDDPM_main.py.py --n 2 --na 1 --T 20 --Ndata 100 --Encode_type "Rx-pi" --batch_size 4 --L $l --seed $s  --lr 0.0005 --weight_t0 0.01 --problem "cluster" --loss_type "MMD"
#done
#done

#python TQuDDPM_main.py.py --n 2 --na 1 --T 20 --Ndata 100 --Encode_type "Ry-pi" --batch_size 4 --L 10 --seed 1  --lr 0.0005 --weight_t0 0.01 --problem "cluster" --loss_type "MMD"
#python TQuDDPM_main.py.py --n 1 --na 2 --T 40 --Ndata 500 --Encode_type "Ry-pi" --batch_size 4 --L 10 --seed 1  --lr 0.0001 --weight_t0 0.05 --problem "circle" --loss_type "Wasserstein"
#python TQuDDPM_main.py.py --n 1 --na 2 --T 40 --Ndata 500 --Encode_type "Ry-pi" --batch_size 4 --L 10 --seed 1  --lr 0.0001 --weight_t0 0.1 --problem "circle" --loss_type "Wasserstein"
#python TQuDDPM_main.py.py --n 1 --na 2 --T 40 --Ndata 500 --Encode_type "Ry-pi" --batch_size 4 --L 10 --seed 1  --lr 0.0001 --weight_t0 0.5 --problem "circle" --loss_type "Wasserstein"
#python TQuDDPM_main.py.py --n 1 --na 2 --T 40 --Ndata 500 --Encode_type "Ry-pi" --batch_size 4 --L 10 --seed 1  --lr 0.0001 --weight_t0 0.2 --problem "circle" --loss_type "Wasserstein"


#   ---------------------------------------------------------------------------------




#   ---------------------------------------------------------------------------------











#for weight in 1.0 0.8 0.2 0.1 0.08 0.05 0.02 0.01
 #do
 #for s in 1 2 3 4 5
 #do
 #  python TQuDDPM_main.py.py --na 2 --Encode_type 'Ry-pi' --batch_size 4 --L 10 --seed $s  --lr 0.0001 --weight_t0 $weight
 #done
 #done

#
#activate QuDDPM
# python TQuDDPM_main.py.py --na 2 --Encode_type "Embedding" --batch_size 4 --L 12 --seed 1  --lr 0.0001 --L_tau 1
# python TQuDDPM_main.py.py --na 2 --Encode_type "Embedding" --batch_size 4 --L 12 --seed 1  --lr 0.0001 --L_tau 2
# python TQuDDPM_main.py.py --na 2 --Encode_type "Embedding" --batch_size 4 --L 12 --seed 1  --lr 0.0001 --L_tau 3
# python TQuDDPM_main.py.py --na 2 --Encode_type "Embedding" --batch_size 4 --L 12 --seed 1  --lr 0.0001 --L_tau 4
# python TQuDDPM_main.py.py --na 2 --Encode_type "Embedding" --batch_size 4 --L 12 --seed 1  --lr 0.0001 --L_tau 5
# python TQuDDPM_main.py.py --na 2 --Encode_type "Embedding" --batch_size 4 --L 12 --seed 1  --lr 0.0001 --L_tau 6
# python TQuDDPM_main.py.py --na 2 --Encode_type "Embedding" --batch_size 4 --L 12 --seed 1  --lr 0.0001 --L_tau 7
# python TQuDDPM_main.py.py --na 2 --Encode_type "Embedding" --batch_size 4 --L 12 --seed 1  --lr 0.0001 --L_tau 8

#for s in 1 2 3 4
#do
#for type in "Ry-pi"
#do
#  python TQuDDPM_main.py.py --na 2 --Encode_type $type --batch_size 4 --L 12 --seed $s  --lr 0.0002 --epoch 30000
#done
#done
#
#for s in 1 2 3 4
#do
#for type in "Ry-2pi"
#do
#  python TQuDDPM_main.py.py --na 2 --Encode_type $type --batch_size 4 --L 12 --seed $s  --lr 0.0001 --epoch 30000
#done
#done
#
#for s in 1 2 3 4
#do
#for type in "Ry-2pi"
#do
#  python TQuDDPM_main.py.py --na 2 --Encode_type $type --batch_size 4 --L 12 --seed $s  --lr 0.0002 --epoch 30000
#done
#done
#
#for s in 5
#do
#python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-pi" --batch_size 4 --L 12 --seed $s  --lr 0.0001 --epoch 30000
#python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-2pi" --batch_size 4 --L 12 --seed $s  --lr 0.0001 --epoch 30000
#python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-pi" --batch_size 4 --L 12 --seed $s  --lr 0.0002 --epoch 30000
#python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-2pi" --batch_size 4 --L 12 --seed $s  --lr 0.0002 --epoch 30000
#done

# python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-pi" --batch_size 4 --L 12 --seed 1  --lr 0.0004
# python TQuDDPM_main.py.py --na 2 --Encode_type "Ry-2pi" --batch_size 4 --L 12 --seed 1  --lr 0.0004
