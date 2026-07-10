import argparse

from src.experiment import ModelRunSpec, run_model_suite

from .common import add_common_arguments, configured_preset


def main():
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Run attention-model ablations.")
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "direct_cross_attention",
            "conservative_direct_cross_attention",
            "residual_cross_attention",
            "gated_residual_cross_attention",
        ],
    )
    args = parser.parse_args()

    preset_by_model = {
        "direct_cross_attention": "attention_balanced",
        "conservative_direct_cross_attention": "conservative_final",
        "residual_cross_attention": "candidate_a",
        "gated_residual_cross_attention": "candidate_a",
        "late_fusion": "candidate_a",
    }
    specs = []
    for seed in args.seeds:
        for model_name in args.models:
            if model_name not in preset_by_model:
                raise ValueError(f"Unsupported attention model: {model_name}")
            config = configured_preset(preset_by_model[model_name], seed, args)
            specs.append(ModelRunSpec(model_name, config))

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
