from src.problem import Problem
from argparse import ArgumentParser

if __name__ == '__main__':
    argument_parser = ArgumentParser(description='Load an IDARP instance and print its summary.')
    argument_parser.add_argument('folder', type=str, help='Path to the instance folder')
    
    args = argument_parser.parse_args()

    problem = Problem(args.folder)
    
    print(problem)