import os

FILENAME_ATTRS_TO_EXCLUDE = [
    "evaluation_objectives",
    "output_dir",
    "logging_dir",
    "overwrite_output_dir",
    "do_train",
    "do_eval",
    "per_device_eval_batch_size",
    "logging_strategy",
    "logging_first_step",
    "logging_steps",
    "save_strategy",
    "eval_steps",
    "label_names",
    "report_to",
    "evaluation_strategy",
    "mteb_eval_tasks",
    "task_to_instructions_fps",
    "mteb_eval_batch_size",
    "eval_pooling_strategy",
    "autoencoding_tokens_for_symmetric_tasks",
    "autoencoding_tokens_for_y_in_symmetric_tasks",
    "autoencoding_tokens_for_query",
    "wandb_run_group",
]


def save_args_to_yaml(args, output_dir, name="run_config.yml"):
    args_dict = args if isinstance(args, dict) else args.to_dict()
    with open(os.path.join(output_dir, name), "a") as f:
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
