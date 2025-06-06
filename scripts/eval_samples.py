from pathlib import Path
import pickle

def main():
    path = Path("train_prep/")
    problems = 0
    for sample_file in path.glob("*.pkl"):
        with open(sample_file, 'rb') as f:
            sample = pickle.load(f)
        
        if len("".join(sample.sequences)) != len(sample.matrix):
            print(f"Sample {sample_file.stem} is invalid: total sequence length {len(''.join(sample.sequences))} does not match matrix size {len(sample.matrix)}.")
            print(f"Sample sequences: {sample.sequences}")
            problems += 1
            # remove the sample file
            sample_file.unlink()
            continue
    print(f"Total problems found: {problems}")
if __name__ == "__main__":
    main()