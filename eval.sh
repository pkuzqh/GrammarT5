dataset=concode
number=8
python3 sptest2.py $dataset $number
python3 palrun.py $number $dataset
python3 sum.py $number java $dataset
python3 testbleu.py $dataset
#cp out.txt base-res/$dataset.txt