# #!/bin/bash

# for seed in 45;
# do
#     bash ./train_baseline1.sh $seed
# done

# #!/bin/bash
# for seed in 42 43 44 45;
# do
#     bash ./train_baseline2.sh $seed
# done

# #!/bin/bash
# for seed in 42 43 44 45;
# do
#     bash ./train_dual.sh $seed
# done

# #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     ./train_lowerbound.sh $seed
# done

#!/bin/bash
for seed in 41 42 43 44 45;
do
    bash ./train_mmbind.sh $seed 0
done


# #!/bin/bash
# for seed in 41 42 43 44 45;
# do
#     bash ./train_upperbound.sh $seed
# done