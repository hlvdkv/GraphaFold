import click

@click.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the input *.dot file",
    required=True,
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=True),
    help="Path to the output dir, where the prediction results will be saved",
    required=True,
)
def main(input: str, output: str) -> None:
    """
    Main function to run the GraphAFold prediction.
    """
    print(f"Input file: {input}")
    print(f"Output directory: {output}")
    # TODO: load the input file
    # TODO: parse to graph
    # TODO: run the prediction
    # TODO: save the prediction results