import argparse

from src.experiment import ModelRunSpec, run_model_suite

from .common import add_common_arguments, configured_preset


def main():
    parser = add_common_arguments(
        argparse.ArgumentParser(
            description="Run the final proposed models and strong MLP baselines."
        )
    )
    args = parser.parse_args()
    specs = []
    for seed in args.seeds:
        specs.extend(
            [
                ModelRunSpec(
                    "conservative_direct_cross_attention",
                    configured_preset("conservative_final", seed, args),
                    "conservative",
                ),
                ModelRunSpec(
                    "direct_cross_attention",
                    configured_preset("attention_balanced", seed, args),
                    "direct",
                ),
                ModelRunSpec(
                    "concat_mlp",
                    configured_preset("candidate_a", seed, args),
                    "concat",
                ),
                ModelRunSpec(
                    "technical_only_mlp",
                    configured_preset("candidate_a", seed, args),
                    "technical",
                ),
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
