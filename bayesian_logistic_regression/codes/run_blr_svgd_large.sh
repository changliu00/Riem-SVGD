for i in `seq $1`
do
	echo 'blr_svgd running round' $i
	python blr_svgd_large.py
done
