# #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_baseline1.sh $seed
# done
# !/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_baseline2.sh $seed
# done
# !/bin/bash
# !/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_baseline3.sh $seed
# done
# !/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_baseline4.sh $seed
# done
# for seed in 41 42 43 44 45;
# do
#     ./train_dual_contrastive.sh $seed
# done
# !/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_lowerbound.sh $seed
# done
#!/bin/bash
for seed in 41 42 43 44 45;
do
    ./train_upperbound.sh $seed
done

# for seed in 41 42 43 44 45;
# do
#     ./train_mmbind_incomplete.sh $seed
# done