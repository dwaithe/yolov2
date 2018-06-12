import os, sys, argparse
import numpy as np
import _pickle as pickle
import datetime

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='list results')
    parser.add_argument('iterations', nargs=1, help='iterations',type=str)
    parser.add_argument('outpath', nargs=1, help='out path for log', type=str)
    
    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    
    file_to_open = args.iterations[0]
    outpath = args.outpath[0]
    b = pickle.load(open(file_to_open,'rb'))

    out = str(datetime.datetime.now())+'\t'+str(file_to_open)+'\titerations\t'+str(file_to_open.split('_')[-2])+'\tmAP\t'+str(b['ap'])
    
    file_name = 'my_file.txt'
    f = open(outpath+'log.txt', 'a+')  # open file in append mode
    print("saving file to:",outpath+'log.txt')
    f.write(out+'\n')
    f.close()