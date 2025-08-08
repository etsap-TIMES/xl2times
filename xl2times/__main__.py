import sys

from .main import parse_args, run


# Python requires a function called `main` in this file
def main(arg_list: None | list[str] = None) -> None:
    """Main entry point for the xl2times tool.

    Returns
    -------
        None.
    """
    args = parse_args(arg_list)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
