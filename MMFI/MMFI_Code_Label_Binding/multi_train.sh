# # #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_baseline1.sh $seed
# done

# for seed in 41 42 43 44 45;
# do
#     ./train_baseline2.sh $seed
# done

# for seed in 41 42 43 44 45;
# do
#     ./train_baseline3.sh $seed
# done

for seed in 41 42 43 44 45;
do
    ./train_mmbind.sh $seed
done

# for seed in 41 42 43 44 45;
# do
#     ./train_upper_bound.sh $seed
# done
