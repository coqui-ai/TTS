class TrainerCallback:
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_init_start(self) -> None:
        if hasattr(self.trainer.model, "on_init_start"):
            self.trainer.model.on_init_start(self.trainer)

        if hasattr(self.trainer.criterion, "on_init_start"):
            self.trainer.criterion.on_init_start(self.trainer)

        if hasattr(self.trainer.optimizer, "on_init_start"):
            self.trainer.optimizer.on_init_start(self.trainer)

    def on_init_end(self) -> None:
        if hasattr(self.trainer.model, "on_init_end"):
            self.trainer.model.on_init_end(self.trainer)

        if hasattr(self.trainer.criterion, "on_init_end"):
            self.trainer.criterion.on_init_end(self.trainer)

        if hasattr(self.trainer.optimizer, "on_init_end"):
            self.trainer.optimizer.on_init_end(self.trainer)

    def on_epoch_start(self) -> None:
        if hasattr(self.trainer.model, "on_epoch_start"):
            self.trainer.model.on_epoch_start(self.trainer)

        if hasattr(self.trainer.criterion, "on_epoch_start"):
            self.trainer.criterion.on_epoch_start(self.trainer)

        if hasattr(self.trainer.optimizer, "on_epoch_start"):
            self.trainer.optimizer.on_epoch_start(self.trainer)

    def on_epoch_end(self) -> None:
        if hasattr(self.trainer.model, "on_epoch_end"):
            self.trainer.model.on_epoch_end(self.trainer)

        if hasattr(self.trainer.criterion, "on_epoch_end"):
            self.trainer.criterion.on_epoch_end(self.trainer)

        if hasattr(self.trainer.optimizer, "on_epoch_end"):
            self.trainer.optimizer.on_epoch_end(self.trainer)

    def on_train_step_start(self) -> None:
        if hasattr(self.trainer.model, "on_train_step_start"):
            self.trainer.model.on_train_step_start(self.trainer)

        if hasattr(self.trainer.criterion, "on_train_step_start"):
            self.trainer.criterion.on_train_step_start(self.trainer)

        if hasattr(self.trainer.optimizer, "on_train_step_start"):
            self.trainer.optimizer.on_train_step_start(self.trainer)

    def on_train_step_end(self) -> None:

        if hasattr(self.trainer.model, "on_train_step_end"):
            self.trainer.model.on_train_step_end(self.trainer)

        if hasattr(self.trainer.criterion, "on_train_step_end"):
            self.trainer.criterion.on_train_step_end(self.trainer)

        if hasattr(self.trainer.optimizer, "on_train_step_end"):
            self.trainer.optimizer.on_train_step_end(self.trainer)

    def on_keyboard_interrupt(self) -> None:
        if hasattr(self.trainer.model, "on_keyboard_interrupt"):
            self.trainer.model.on_keyboard_interrupt(self.trainer)

        if hasattr(self.trainer.criterion, "on_keyboard_interrupt"):
            self.trainer.criterion.on_keyboard_interrupt(self.trainer)

        if hasattr(self.trainer.optimizer, "on_keyboard_interrupt"):
            self.trainer.optimizer.on_keyboard_interrupt(self.trainer)
