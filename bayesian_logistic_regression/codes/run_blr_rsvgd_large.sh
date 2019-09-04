for i in `seq $1`
do
	echo 'blr_rsvgd running round' $i
	python blr_rsvgd_large.py
done
