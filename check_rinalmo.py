import json

from graphafold.diagnostics import check_rinalmo


def main() -> None:
    print(json.dumps(check_rinalmo(), indent=2))


if __name__ == "__main__":
    main()
