from TTS.utils.logging.console_logger import ConsoleLogger
from TTS.utils.logging.tensorboard_logger import TensorboardLogger
from TTS.utils.logging.wandb_logger import WandbLogger


def init_dashboard_logger(config):
    if config.dashboard_logger == "tensorboard":
        dashboard_logger = TensorboardLogger(config.output_log_path, model_name=config.model)

    elif config.dashboard_logger == "wandb":
        project_name = config.model
        if config.project_name:
            project_name = config.project_name

        dashboard_logger = WandbLogger(
            project=project_name,
            name=config.run_name,
            config=config,
            entity=config.wandb_entity,
        )

    dashboard_logger.add_text("model-config", f"<pre>{config.to_json()}</pre>", 0)

    return dashboard_logger
