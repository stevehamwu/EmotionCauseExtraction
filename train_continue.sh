#!/bin/bash
SESS=$1
START=$2
END=$3
for ((EXP=$START;EXP<=$END;EXP++))
do
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
