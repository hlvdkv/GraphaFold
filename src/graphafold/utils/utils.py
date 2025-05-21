import os
import tempfile
import numpy as np

def parse_dotbracket(dotbracket):
    """Parse base pairings from dot-bracket notation across the full sequence."""
    stack = []
    pairs = {}
    for i, char in enumerate(dotbracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs

def write_idx_file(sequences, idx_path):
    with open(idx_path, 'w') as f:
        idx = 1
        for chain_id, sequence in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", sequences):
            for i, nt in enumerate(sequence):
                f.write(f"{idx},{chain_id}.{nt}{i+1}\n")
                idx += 1

def write_cmt_file(sequences, dotbrackets, cmt_path):
    # Flatten the sequences and dotbrackets
    full_sequence = ''.join(sequences)
    full_dotbracket = ''.join(dotbrackets)
    total_length = len(full_sequence)

    matrix = np.zeros((total_length, total_length), dtype=int)

    # Base pairs from dot-bracket notation
    pairs = parse_dotbracket(full_dotbracket)
    for i, j in pairs.items():
        if i < j:  # avoid duplicate setting
            matrix[i, j] = 1
            matrix[j, i] = 1

    # Covalent bonds *within chains only*
    offset = 0
    for seq in sequences:
        for i in range(len(seq) - 1):
            matrix[offset + i, offset + i + 1] = -1
            matrix[offset + i + 1, offset + i] = -1
        offset += len(seq)

    # Save matrix to file
    with open(cmt_path, 'w') as f:
        for row in matrix:
            f.write(','.join(map(str, row)) + '\n')

def parse_input_file(input_file_path):
    with open(input_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        sequences = lines[0].split(';')
        dotbrackets = lines[1].split(';')
        return sequences, dotbrackets

def parse_dot2out(input_file_path):
    tmpdir = tempfile.mkdtemp()
    idx_path = os.path.join(tmpdir, "tmp.idx")
    cmt_path = os.path.join(tmpdir, "tmp.cmt")

    sequences, dotbrackets = parse_input_file(input_file_path)

    write_idx_file(sequences, idx_path)
    write_cmt_file(sequences, dotbrackets, cmt_path)

    print(f"Files written to {tmpdir}")
    return idx_path, cmt_path

if __name__ == "__main__":
    import sys
    parse_dot2out(sys.argv[1])
# Example usage:
# idx_file, cmt_file = main("path/to/input.txt")
