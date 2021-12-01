class TrainerCallback:
    @staticmethod
    def on_init_start(trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_init_start"):
                trainer.model.module.on_init_start(trainer)
        else:
            if hasattr(trainer.model, "on_init_start"):
                trainer.model.on_init_start(trainer)

        if hasattr(trainer.criterion, "on_init_start"):
            trainer.criterion.on_init_start(trainer)

        if hasattr(trainer.optimizer, "on_init_start"):
            trainer.optimizer.on_init_start(trainer)

    @staticmethod
    def on_init_end(trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_init_end"):
                trainer.model.module.on_init_end(trainer)
        else:
            if hasattr(trainer.model, "on_init_end"):
                trainer.model.on_init_end(trainer)

        if hasattr(trainer.criterion, "on_init_end"):
            trainer.criterion.on_init_end(trainer)

        if hasattr(trainer.optimizer, "on_init_end"):
            trainer.optimizer.on_init_end(trainer)

    @staticmethod
    def on_epoch_start(trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_epoch_start"):
                trainer.model.module.on_epoch_start(trainer)
        else:
            if hasattr(trainer.model, "on_epoch_start"):
                trainer.model.on_epoch_start(trainer)

        if hasattr(trainer.criterion, "on_epoch_start"):
            trainer.criterion.on_epoch_start(trainer)

        if hasattr(trainer.optimizer, "on_epoch_start"):
            trainer.optimizer.on_epoch_start(trainer)

    @staticmethod
    def on_epoch_end(trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_epoch_end"):
                trainer.model.module.on_epoch_end(trainer)
        else:
            if hasattr(trainer.model, "on_epoch_end"):
                trainer.model.on_epoch_end(trainer)

        if hasattr(trainer.criterion, "on_epoch_end"):
            trainer.criterion.on_epoch_end(trainer)

        if hasattr(trainer.optimizer, "on_epoch_end"):
            trainer.optimizer.on_epoch_end(trainer)

    @staticmethod
    def on_train_step_start(trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_train_step_start"):
                trainer.model.module.on_train_step_start(trainer)
        else:
            if hasattr(trainer.model, "on_train_step_start"):
                trainer.model.on_train_step_start(trainer)

        if hasattr(trainer.criterion, "on_train_step_start"):
            trainer.criterion.on_train_step_start(trainer)

        if hasattr(trainer.optimizer, "on_train_step_start"):
            trainer.optimizer.on_train_step_start(trainer)

    @staticmethod
    def on_train_step_end(trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_train_step_end"):
                trainer.model.module.on_train_step_end(trainer)
        else:
            if hasattr(trainer.model, "on_train_step_end"):
                trainer.model.on_train_step_end(trainer)

        if hasattr(trainer.criterion, "on_train_step_end"):
            trainer.criterion.on_train_step_end(trainer)

        if hasattr(trainer.optimizer, "on_train_step_end"):
            trainer.optimizer.on_train_step_end(trainer)

    @staticmethod
    def on_keyboard_interrupt(trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_keyboard_interrupt"):
                trainer.model.module.on_keyboard_interrupt(trainer)
        else:
            if hasattr(trainer.model, "on_keyboard_interrupt"):
                trainer.model.on_keyboard_interrupt(trainer)

        if hasattr(trainer.criterion, "on_keyboard_interrupt"):
            trainer.criterion.on_keyboard_interrupt(trainer)

        if hasattr(trainer.optimizer, "on_keyboard_interrupt"):
            trainer.optimizer.on_keyboard_interrupt(trainer)
