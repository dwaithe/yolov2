import os, sys, argparse
import numpy as np
import _pickle as pickle

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='list results')
    parser.add_argument('iterations', nargs=1, help='iterations',type=str)
    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    
    iterations = args.iterations[0]
    b = pickle.load(open('results/cell - neuroblastoma phalloidin_'+iterations+'_pr.pkl','rb'))

    print('iterations\t'+str(iterations)+'\tmAP\t'+str(b['ap']))

    
