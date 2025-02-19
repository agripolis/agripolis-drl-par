resdir=RES
for i in {0..4}
do
  echo ${resdir} $i
  python mlu/train.py ${resdir} $i > r$i.res
  mv r$i.res outputfiles* ${resdir}/$i
done
