from argparse import ArgumentParser

from src.problem import Problem

from src.construction import construct
from src.neighbors import expand
from src.solver import solve

if __name__ == '__main__':
    argument_parser = ArgumentParser(description='Load an IDARP instance and print its summary.')
    argument_parser.add_argument('folder', type=str, help='Path to the instance folder')
    
    args = argument_parser.parse_args()

    print('Loading problem...')
    problem = Problem(args.folder)
    
    print('Initial solution...')
    initial_solution = construct(problem)
    print('Validating initial solution')
    initial_solution.validate()
    
    print('Expanding neighbors...')
    matrices = expand(problem, initial_solution, k=1, k_outside=1)
    
    print('Improving solution...')
    improved_solution = solve(problem, initial_solution, matrices)
    print('Validating improved solution...')
    improved_solution.validate()
    
    print('Initial Cost:', initial_solution.cost)
    print('Improved Cost:', improved_solution.cost)