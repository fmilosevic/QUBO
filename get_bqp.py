import numpy as np
import os

def get_bqp(file_path, problem_index=0):
    """
    Parse a BQP50 format file and return the Q matrix for the specified problem index.
    
    Args:
        file_path (str): Path to the BQP50 file
        problem_index (int): Index of the problem to load (0-based)
        
    Returns:
        tuple: (n, Q) where n is the number of variables and Q is the matrix
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse number of test problems
        num_problems = int(lines[0].strip())
        if problem_index >= num_problems:
            raise ValueError(f"Problem index {problem_index} out of range (there are {num_problems} problems)")
        
        # Skip to the problem we want
        line_index = 1
        for i in range(problem_index):
            # Read problem header (n and nnz)
            n, nnz = map(int, lines[line_index].strip().split())
            # Skip the non-zero elements
            line_index += nnz + 1
        
        # Now we're at the problem we want
        n, nnz = map(int, lines[line_index].strip().split())
        line_index += 1
        
        # Initialize Q matrix
        Q = np.zeros((n, n))
        
        # Read non-zero elements
        for _ in range(nnz):
            i, j, q_val = map(float, lines[line_index].strip().split())
            i, j = int(i) - 1, int(j) - 1  # Convert to 0-based indexing
            Q[i, j] = q_val
            if i != j:  # Ensure symmetry
                Q[j, i] = q_val
            line_index += 1
        
        return n, Q
    except Exception as e:
        print(f"Error parsing file: {e}")
        raise

def get_problem_count(file_path):
    """
    Get the number of problems in a BQP50 file.
    
    Args:
        file_path (str): Path to the BQP50 file
        
    Returns:
        int: Number of problems in the file
    """
    try:
        with open(file_path, 'r') as f:
            return int(f.readline().strip())
    except Exception as e:
        print(f"Error reading file: {e}")
        raise

if __name__ == "__main__":
    # Test the parsing function with a file path
    import sys

    file_path = (r"E:\bac\DIP\test_problems\bqp2500.txt")
    
    
    if not os.path.exists(file_path):
        print(f"Error: File does not exist at {file_path}")
        sys.exit(1)
    
    try:
        num_problems = get_problem_count(file_path)
        print(f"Found {num_problems} problems in the file")
        
        problem_index = 0
        if len(sys.argv) > 2:
            problem_index = int(sys.argv[2])
        else:
            problem_index = int(input(f"Enter problem index (0-{num_problems-1}): "))
        
        n, Q = get_bqp(file_path, problem_index)
        print(f"Problem {problem_index} has {n} variables")
        print("Q matrix:")
        np.set_printoptions(threshold=np.inf)  # Set no limit on printing
        print(Q)
        np.set_printoptions(threshold=1000)  # Back to a reasonable default
        
        # Print some statistics about the Q matrix
        #nonzero = np.count_nonzero(Q)
        #density = nonzero / (n * n)
        #print(f"Matrix density: {density:.4f} ({nonzero} non-zero elements out of {n*n})")
        #print(f"Min value: {np.min(Q)}")
        #print(f"Max value: {np.max(Q)}")
        #print(f"Mean value of non-zero elements: {np.mean(Q[Q != 0]):.4f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")