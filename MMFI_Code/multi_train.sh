#!/bin/bash
for seed in 41;
do
    ./train_baseline2.sh $seed
done
#!/bin/bash
for seed in 41;
do
    ./train_dual_contrastive.sh $seed
done
#!/bin/bash
for seed in 41;
do
    ./train_lowerbound.sh $seed
done
#!/bin/bash
for seed in 41;
do
    ./train_mmbind.sh $seed
done
#!/bin/bash
for seed in 41;
do
    ./train_upperbound.sh $seed
done