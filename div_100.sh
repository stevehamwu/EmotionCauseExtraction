for EXP in {1..100}
do
	python data_process/2_divide.py \
	--session static \
	--exp $EXP
done
