batch_size=10

export THEANO_FLAGS='gpuarray.preallocate=0.05'

python ./fifty_layer_ibis_test.py -l $1 -t
for ((i=0;i<500;i+=$batch_size*3));
do
  echo "Starting batch $i"
  python ./fifty_layer_ibis_test.py -l $1 -b $i -n 10 -r &
  let a=i+batch_size
  echo "Starting batch $a"
  python ./fifty_layer_ibis_test.py -l $1 -b $a -n 10 -r &
  let a=a+batch_size
  echo "Starting batch $a"
  python ./fifty_layer_ibis_test.py -l $1 -b $a -n 10 -r &
  wait
done
