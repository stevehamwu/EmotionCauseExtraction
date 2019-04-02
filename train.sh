#!/bin/bash
SESS=$1

for EXP in {1..25}
do
#    python data_process/2_divide.py \
#    --session $SESS \
#    --exp $EXP
    ipython main.py \
    --pdb \
    -- \
    --session $SESS.$EXP \
    --exp $EXP \
    --config ./protos/$SESS.yaml \
    --curr_time $EXP \
    --ckpt_out $SESS.$EXP \
    --model_out $SESS.$EXP
done
