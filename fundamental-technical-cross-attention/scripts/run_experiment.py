import argparse

from src.experiment import ModelRunSpec, run_model_suite
from src.models import MODEL_NAMES

from .common import add_common_arguments, configured_preset


def main():
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Run a custom model suite.")
    )
    parser.add_argument("--preset", default="candidate_a")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, required=True)
    args = parser.parse_args()
    specs = [
        ModelRunSpec(
            model_name,
            configured_preset(args.preset, seed, args),
        )
        for seed in args.seeds
        for model_name in args.models
    ]
    run_model_suite(
        args.data_path,
        args.output_dir,
        specs,
        n_rows=args.n_rows,
        sample_seed=args.sample_seed,
        evaluate_test=args.evaluate_test,
        save_models=not args.no_save_models,
    )


if __name__ == "__main__":
    main()
