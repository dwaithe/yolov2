import subprocess
import shutil
import sys
import _pickle as pickle
sys.path.append("scripts")
import reval_custom_py3 as revc
import reval_final_list as revf
import datetime


#Global variables
path_to_amca_config = "/home/molimm2/dwaithe/keras_experiments/amca/config/"
path_to_darknet = "/home/molimm2/dwaithe/keras_experiments/darknet2/"
path_to_data = "/home/molimm2/dwaithe/keras_experiments/Faster-RCNN-TensorFlow-Python3.5/data/"
models_path_on_server = "/scratch/dwaithe/models/darknet/"

def get_experiment_parameters(exp_id):
	f = open(path_to_amca_config+'experiment_spec_'+exp_id+'.txt')
	lines = f.readlines()
	model_array =[]
	train_array =[]
	test_array = []
	for line in lines:
		if line[0:19] == "darknet_model_init=":
			model_array.append(line[19:])
		if line[0:17] == "train_on_dataset=":
			train_array.append(line[17:])
		if line[0:16] == "test_on_dataset=":
			test_array.append(line[16:])
		






#local variables
#path_to_darknet ="/Users/dwaithe/Documents/collaborators/WaitheD/micro_vision/darknet/"
#path_to_data = "/Users/dwaithe/Documents/collaborators/WaitheD/micro_vision/Faster-RCNN-TensorFlow-Python3.5/data/"


darknet_model_init = get_experiment_parameters('01'):


path_to_init_model = path_to_darknet+darknet_model_init

#Experiment specific.
num_of_train = '180'
year = '2018'
test_set="test_n"+num_of_train
cell_class="cell - neuroblastoma phalloidin dapi"
dataset = "neuroblastoma_phal_dapi_class"
dataset_dir=dataset+"/"+year+"/"
dn_mixed_class_name=dataset+num_of_train+".data"
path_to_training_def = path_to_data+dataset_dir+"obj_"+dn_mixed_class_name



#Evaluation specific.
#eval_dataset = "neuroblastoma_phal_class"
#eval_dataset_dir=eval_dataset+"/"+year+"/"
#eval_dn_mixed_class_name=eval_dataset+num_of_train+".data"
#eval_path_to_training_def = path_to_data+eval_dataset_dir+"obj_"+eval_dn_mixed_class_name
eval_dataset = dataset
eval_dataset_dir=dataset_dir
eval_dn_mixed_class_name=dn_mixed_class_name
eval_path_to_training_def = path_to_training_def



GPU_to_use = 0
models_itr_to_test =[100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000];
#models_itr_to_test = [10000]

foutfile="indflip_colour_on_noncol"
cfg_file="yolo-obj"

#Training.
#out = subprocess.call(path_to_darknet+"darknet detector train "+path_to_training_def+" "+path_to_darknet+"cfg/"+cfg_file+".cfg "+path_to_init_model+" -gpus "+str(GPU_to_use), shell=True)


#Prediction and Evaluation
for i in models_itr_to_test:
	weights = models_path_on_server+dataset+num_of_train+"/"+cfg_file+"_"+str(i)+".weights"
	out = subprocess.call(path_to_darknet+"darknet detector valid "+eval_path_to_training_def+" "+path_to_darknet+"cfg/"+cfg_file+".cfg "+weights, shell=True)
	print("finished",out)
	inputname = path_to_darknet+"results/comp4_det_test_"+cell_class+".txt" 
	outputname = path_to_darknet+"results/comp4_det_"+test_set+"_"+cell_class+".txt"
	shutil.move(inputname,outputname)
	with open(path_to_data+eval_dataset_dir+"obj_"+eval_dataset+'.names', 'r') as f:
		lines = f.readlines()

	classes = [t.strip('\n') for t in lines]
	
	revc.do_python_eval(path_to_data+eval_dataset, year, test_set,classes , path_to_darknet+"results/", str(i))
	finalname = path_to_darknet+"results/comp4_det_"+test_set+"_"+cell_class+"_"+str(i)+".txt"
	shutil.move(outputname,finalname)



#Printing and collation.
output_path = models_path_on_server+"_"+foutfile
f = open(output_path+'log.txt', 'a+')  # open file in append mode
for i in models_itr_to_test:
	pick_to_open = path_to_darknet+"results/"+cell_class+"_"+str(i)+"_pr.pkl"
	data = pickle.load(open(pick_to_open,'rb'))
	out = str(datetime.datetime.now())+'\t'+str(pick_to_open)+'\titerations\t'+str(pick_to_open.split('_')[-2])+'\tmAP\t'+str(data['ap'])
	print("saving file to:",output_path+'log.txt')
	f.write(out+'\n')
f.close()
