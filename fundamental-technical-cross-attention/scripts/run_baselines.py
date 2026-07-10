import argparse

from src.experiment import ModelRunSpec, run_model_suite

from .common import add_common_arguments, configured_preset


def main():
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Run MLP baselines.")
    )
    args = parser.parse_args()
    specs = []
    for seed in args.seeds:
        config = configured_preset("candidate_a", seed, args)
        specs.extend(
            [
                ModelRunSpec("technical_only_mlp", config, "technical"),
                ModelRunSpec("fundamental_only_mlp", config, "fundamental"),
                ModelRunSpec("concat_mlp", config, "concat"),
            ]
        )
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
