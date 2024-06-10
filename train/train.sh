#!/bin/bash

for seed in 41 42 43 44 45;
do
    bash ./train_baseline1.sh $seed
done

# #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_baseline2.sh $seed
# done
# #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_dual.sh $seed
# done
# #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_lowerbound.sh $seed
# done
# #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_mmbind.sh $seed
# done
# #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_upperbound.sh $seed
# done