resdir=Large/RES
for i in {0..4}
do
  echo ${resdir} $i
  python mlu/Tmain.py ${resdir} $i  #> r$i.res
  mv outputfiles* ${resdir}/$i
done
