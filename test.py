from argparse import ArgumentParser
from time import time

from src.problem import Problem
from src.pipeline import run

def timer(func):
    ''' Decorator to measure the execution time of a function '''
    
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        
        if isinstance(result, tuple):
            return end - start, *result
        
        return end - start, result
    
    return wrapper
    
if __name__ == '__main__':
    argument_parser = ArgumentParser(description='Load an IDARP instance and print its summary.')
    argument_parser.add_argument('folder', type=str, help='Path to the instance folder')
    
    args = argument_parser.parse_args()

    print('Loading problem...')
    problem = Problem(args.folder)
    
    print('Running pipeline...')
    run(problem)