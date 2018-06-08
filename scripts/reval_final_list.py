import os, sys, argparse
import numpy as np
import _pickle as pickle

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='list results')
    parser.add_argument('iterations', nargs=1, help='iterations',type=str)
    parser.add_argument('outpath', nargs=1 help='out path for log', type=str)
    
    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    
    file_to_open = args.iterations[0]
    outpath = args.iterations[1]
    b = pickle.load(open(file_to_open,'rb'))

    out = 'iterations\t'+str(file_to_open.split('_')[-2])+'\tmAP\t'+str(b['ap'])
    print(out)
    file_name = 'my_file.txt'
    f = open(outpath+'log.txt', 'a+')  # open file in append mode
    f.write(out+'\n')
    f.close()

'''
w  write mode
r  read mode
a  append mode

w+  create file if it doesn't exist and open it in write mode
r+  open an existing file in read+write mode
a+  create file if it doesn't exist and open it in append mode
'''''' w  wr
example:



    
