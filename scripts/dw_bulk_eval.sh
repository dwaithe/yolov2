## declare an array variable
declare -a arr=(100,200,1000,10000);





## now loop through the above array
for i in "${arr[@]}"
do
	echo "$i";
	.././darknet detector valid ../cfg/obj.data cfg/yolo-obj.cfg /scratch/dwaithe/models/darknet/MP6843phal_class/yolo-obj_$i.weights;
	mv "/home/molimm2/dwaithe/keras_experiments/darknet/results/comp4_det_test_cell - neuroblastoma phalloidin.txt" "/home/molimm2/dwaithe/keras_experiments/darknet/results/comp4_det_test_n180_cell - neuroblastoma phalloidin.txt";
	python reval_custom_py3.py /home/molimm2/dwaithe/keras_experiments/darknet/results --dataset_dir /home/molimm2/dwaithe/keras_experiments/Faster-RCNN-TensorFlow-Python3.5/data/MP6843phal_class --year 2018 --image_set test_n180 --classes /home/molimm2/dwaithe/keras_experiments/darknet/cfg/obj_neuroblastoma_phal.names;
	mv "/home/molimm2/dwaithe/keras_experiments/darknet/results/comp4_det_test_n180_cell - neuroblastoma phalloidin.txt" "/home/molimm2/dwaithe/keras_experiments/darknet/results/comp4_det_test_n180_cell - neuroblastoma phalloidin_"$i".txt";
done