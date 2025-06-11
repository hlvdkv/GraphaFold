import click
from pathlib import Path
import numpy as np
from graphafold.data import load_matrix, load_idx_file, Sample

@click.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, dir_okay=True),
    help="Path to the input directory containing the data files amt, cmt and idx dirs",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=True),
    default="./output",
    help="Path to the output directory, where the processed data will be saved",
)
def main(input, output):
    """
    Main function to run the GraphAFold preprocessing.
    This function will read the input file and save the processed data to the output directory.
    """
    if not input:
        raise ValueError("Input file must be provided.")
    
    print(f"Input file: {input}")
    print(f"Output directory: {output}")

    amt_dir = Path(input) / "amt"
    idx_dir = Path(input) / "idx"
    
    # Load the matrix from the input file
    for amt in amt_dir.glob("*.amt"):
        print(f"Processing file: {amt}")
        idx_file = idx_dir / (amt.stem + ".idx")
        if not idx_file.exists():
            raise FileNotFoundError(f"Index file {idx_file} does not exist.")
        matrix = load_matrix(amt)
        index_nt_dict, neighbours = load_idx_file(idx_file)
        if matrix.shape[0] != len(index_nt_dict):
            # raise ValueError(f"Matrix size {matrix.shape[0]} does not match idx file size {len(index_nt_dict)}")
            print(f"Warning: Matrix size {matrix.shape[0]} does not match idx file size {len(index_nt_dict)}")
            continue
        # assert matrix.shape[0] == len(index_nt_dict), f"Matrix size {matrix.shape[0]} does not match idx file size {len(index_nt_dict)}"
        # b_a_neighbours - flip columns (0, 1) to (1, 0) for neighbours
        b_a_neighbours = np.array([(b, a) for a, b in neighbours])
        neighbours = np.concatenate([neighbours, b_a_neighbours], axis=0)
        sample = Sample(matrix, index_nt_dict, neighbours)
        if not sample.is_valid():
            continue
        # create output directory if it doesn't exist
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save the sample to a pickle file
        sample_name = amt.stem
        sample.save_to_pickle(output_dir / (f"{sample_name}.pkl"))
        print(f"Processed {sample_name} and saved to {output_dir / (sample_name + '.pkl')}")

if __name__ == "__main__":
    main()