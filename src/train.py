import argparse
import os
import sys
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

from dataset.dataset import DeepGate2Collater, PackedSplitDataset
from model.DeepGate2 import DeepGate2
from model.loss import (
    LOSS_TYPE,
    loss_hyperparameters,
    tanh_control_parameters,
    validate_loss_checkpoint,
)
from model.module import (
    TRAINING_MONITOR,
    SECLModule,
)


_EXCLUDED_BATCH_KEYS = {
    "functional_signature",
    "num_base_nodes",
    "circuit_name",
}


class EpochWallTimer(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        print(
            f"[State] rank={trainer.global_rank} epoch={trainer.current_epoch} "
            f"global_step={trainer.global_step} lr={optimizer.param_groups[0]['lr']} "
            f"optimizer_states={len(optimizer.state)}",
            flush=True,
        )

    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero and (batch_idx == 0 or (batch_idx + 1) % 20 == 0):
            loss = outputs.get("loss") if isinstance(outputs, dict) else None
            value = float(loss.detach()) if loss is not None else None
            print(f"[Progress] epoch={trainer.current_epoch} batch={batch_idx + 1}/{trainer.num_training_batches} "
                  f"step={trainer.global_step} elapsed={time.perf_counter() - self.started:.1f}s loss={value}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        elapsed = torch.tensor(time.perf_counter() - self.started, device=pl_module.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
        pl_module.log("epoch_seconds", elapsed, on_step=False, on_epoch=True)
        if trainer.is_global_zero:
            val_loss = trainer.callback_metrics.get("val_loss")
            print(f"[Timing] epoch={trainer.current_epoch} train+val={float(elapsed):.3f}s val_loss={val_loss}", flush=True)


# Also refresh last.ckpt every epoch so an interrupted run can be resumed.
class ResumableModelCheckpoint(pl.callbacks.ModelCheckpoint):
    @property
    def state_key(self):
        key = super().state_key
        return "ModelCheckpoint" + key[len(type(self).__qualname__):]

    def on_train_epoch_end(self, trainer, pl_module):
        eligible = (not self._should_skip_saving_checkpoint(trainer)
                    and self._should_save_on_train_epoch_end(trainer))
        super().on_train_epoch_end(trainer, pl_module)
        if eligible and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_validation_end(self, trainer, pl_module):
        eligible = (not self._should_skip_saving_checkpoint(trainer)
                    and not self._should_save_on_train_epoch_end(trainer))
        super().on_validation_end(trainer, pl_module)
        if eligible and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


def load_packed_datasets(data_dir, *, load_test=True):
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    test_path = os.path.join(data_dir, "test.pt")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Required packed split not found: {train_path}")

    train_dataset = PackedSplitDataset(train_path)
    val_dataset = (
        PackedSplitDataset(val_path) if os.path.isfile(val_path) else None
    )
    test_dataset = (
        PackedSplitDataset(test_path)
        if load_test and os.path.isfile(test_path)
        else None
    )
    if val_dataset is not None and len(val_dataset) == 0:
        val_dataset = None
    if test_dataset is not None and len(test_dataset) == 0:
        test_dataset = None
    return train_dataset, val_dataset, test_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Train PI-aware DeepGate2 with the SECL loss"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/dataset/split",
        help="Directory containing packed train.pt/val.pt/test.pt",
    )

    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument(
        "--deepgate2_num_rounds",
        type=int,
        default=1,
        help="Number of DeepGate2 forward propagation rounds",
    )
    parser.add_argument(
        "--deepgate2_execution", choices=["dag", "dag_reference", "dense"], default="dag",
        help="Propagation backend: dag (GRUCell), dag_reference (nn.GRU) or dense",
    )
    parser.add_argument(
        "--secl_compile", action="store_true",
        help="Compile SECL denominator blocks with torch.compile",
    )
    parser.add_argument(
        "--deepgate2_gate_replay", action="store_true",
        help="Use CUDA graphs for small DeepGate2 levels",
    )
    parser.add_argument("--deepgate2_replay_max_gates", type=int, default=128)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of circuit graphs per device and batch",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches over which to accumulate gradients",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=60,
                        help="Validation patience; 0 disables early stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=["32-true", "bf16-mixed", "16-mixed"],
        default="bf16-mixed",
        help="Lightning precision",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=20,
        help="Training progress logging interval",
    )

    parser.add_argument(
        "--grad_scale_m", type=float, default=2.0,
        help="M in exp(ln(M)*tanh(d/delta)); M >= 1",
    )
    parser.add_argument(
        "--grad_scale_delta", type=float, default=0.1,
        help="delta in exp(ln(M)*tanh(d/delta)); delta > 0",
    )
    parser.add_argument(
        "--singleton_similarity_center", type=float, default=0.8,
        help="Similarity center of the singleton penalty",
    )
    parser.add_argument(
        "--singleton_softplus_width", type=float, default=0.1,
        help="Softplus width of the singleton penalty",
    )
    parser.add_argument(
        "--stage2_temperature",
        type=float,
        default=0.1,
        help="Temperature of circuit-pool member-to-prototype SECL",
    )
    parser.add_argument(
        "--stage2_margin",
        type=float,
        default=0.3,
        help="Signed angular positive margin for prototype SECL",
    )
    parser.add_argument(
        "--stage2_singleton_circle_gamma",
        dest="singleton_circle_gamma",
        type=float,
        default=32.0,
        help="Circle-style logit scale for singleton-pair denominator mass",
    )
    parser.add_argument(
        "--stage2_singleton_negative_weight",
        dest="singleton_negative_weight",
        type=float,
        default=1.0,
        help="Weight of singleton pairs in the prototype-SECL denominator",
    )
    parser.add_argument(
        "--secl_singleton_pair_chunk_size",
        dest="singleton_pair_chunk_size",
        type=int,
        default=1_048_576,
        help="Max singleton pairs per block; larger classes are streamed",
    )
    parser.add_argument(
        "--secl_key_chunk_size",
        "--key_chunk_size",
        dest="key_chunk_size",
        type=int,
        default=65536,
        help="Key chunk size for streamed circuit candidate pools",
    )
    parser.add_argument(
        "--secl_anchor_chunk_size",
        "--anchor_chunk_size",
        dest="anchor_chunk_size",
        type=int,
        default=4096,
        help="Anchor chunk size for streamed circuit candidate pools",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="torch intra-op CPU threads; 0 keeps the torch default",
    )
    parser.add_argument(
        "--secl_checkpoint_free_bytes",
        dest="checkpoint_free_bytes",
        type=int,
        default=0,
        help="Activation memory (bytes) kept before gradient checkpointing; 0 checkpoints all blocks",
    )

    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["gpu", "cpu", "auto"],
        default="gpu",
    )

    parser.add_argument("--checkpoint_dir", default=None,
                        help="Checkpoint directory (default: <log_dir>/checkpoints)")
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--allow_batch_size_change", action="store_true",
                        help="Allow a different batch size when resuming")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results/secl",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Encoder or Lightning checkpoint used to initialize encoder weights",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Lightning checkpoint to resume training from (optimizer and scheduler included)",
    )
    args = parser.parse_args()
    args.encoder_type = "DeepGate2"
    args.deepgate2_pi_init_mode = "const"
    args.loss_type = LOSS_TYPE
    if args.lr is None:
        args.lr = 1e-4
    return args


def make_loader(dataset, args):
    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DeepGate2Collater(sorted(_EXCLUDED_BATCH_KEYS)),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.accelerator != "cpu",
    )


def restore_best_model(model, lightning_checkpoint):
    checkpoint = torch.load(
        lightning_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError("Lightning checkpoint has no state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=True)


def load_encoder(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_config" in payload:
        return DeepGate2.from_checkpoint(checkpoint_path).eval()
    if not isinstance(payload, dict):
        raise RuntimeError("Expected a DeepGate2 or Lightning checkpoint")
    hparams, state = payload.get("hyper_parameters"), payload.get("state_dict")
    if not isinstance(hparams, dict) or not isinstance(state, dict):
        raise RuntimeError("Checkpoint must contain hyper_parameters and state_dict")
    if hparams.get("encoder_type") != "DeepGate2":
        raise RuntimeError("Checkpoint does not contain DeepGate2")
    model = DeepGate2(
        dim_hidden=hparams["hidden"],
        num_rounds=hparams["deepgate2_num_rounds"],
        pi_init_mode=hparams["deepgate2_pi_init_mode"],
    )
    model.load_state_dict({key.removeprefix("encoder."): value
                           for key, value in state.items() if key.startswith("encoder.")},
                          strict=True)
    return model.eval()


def initialize_encoder(model, checkpoint_path):
    encoder = load_encoder(checkpoint_path)
    if type(encoder) is not type(model.encoder) or encoder.model_config != model.encoder.model_config:
        raise RuntimeError("initialization checkpoint encoder configuration differs from this run")
    model.encoder.load_state_dict(encoder.state_dict(), strict=True)


def validate_resume_checkpoint(checkpoint_path, args) -> int:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    required = {
        "epoch",
        "global_step",
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "loops",
        "callbacks",
        "hyper_parameters",
    }
    if not isinstance(checkpoint, dict) or not required <= set(checkpoint):
        raise RuntimeError(
            "--resume_checkpoint must be a full Lightning checkpoint "
            "containing model, optimizer, scheduler, loop, and step state"
        )
    hyper_parameters = checkpoint["hyper_parameters"]
    if not isinstance(hyper_parameters, dict):
        raise RuntimeError(
            "resume checkpoint hyper_parameters must be a mapping"
        )
    if (
        not isinstance(checkpoint["optimizer_states"], list)
        or not checkpoint["optimizer_states"]
        or not isinstance(checkpoint["lr_schedulers"], list)
        or not isinstance(checkpoint["loops"], dict)
        or not isinstance(checkpoint["callbacks"], dict)
    ):
        raise RuntimeError(
            "resume checkpoint does not contain complete Lightning "
            "training state"
        )
    validate_loss_checkpoint(hyper_parameters, loss_hyperparameters(args))
    if hyper_parameters.get("batch_size_scope", "per_device") != "per_device":
        raise RuntimeError(
            "Checkpoint uses a global batch size; use --checkpoint to initialize encoder weights"
        )
    compatible_fields = (
        "encoder_type",
        "hidden",
        "batch_size",
        "accumulate_grad_batches",
        "lr",
        "key_chunk_size",
        "anchor_chunk_size",
        "stage2_temperature",
        "stage2_margin",
        "singleton_circle_gamma",
        "singleton_negative_weight",
        "singleton_pair_chunk_size",
    )
    compatible_fields += ("deepgate2_num_rounds", "deepgate2_pi_init_mode")
    for field in compatible_fields:
        if field == "batch_size" and getattr(args, "allow_batch_size_change", False):
            continue
        expected = getattr(args, field)
        actual = hyper_parameters.get(field)
        if actual != expected:
            raise RuntimeError(
                f"resume checkpoint {field} is incompatible: "
                f"expected {expected!r}, found {actual!r}"
            )
    checkpoint_feature_config = hyper_parameters.get("feature_config")
    if checkpoint_feature_config != getattr(args, "feature_config", None):
        raise RuntimeError(
            "resume checkpoint feature_config is incompatible with the "
            "current packed dataset"
        )
    epoch = checkpoint["epoch"]
    if not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError("resume checkpoint epoch is invalid")
    if args.max_epochs <= epoch + 1:
        raise ValueError(
            f"--max_epochs={args.max_epochs} must exceed the completed "
            f"resume epoch count {epoch + 1}"
        )
    return epoch


def main():
    args = get_args()
    if args.batch_size < 1 or args.max_epochs < 1:
        raise ValueError("batch_size and max_epochs must be positive")
    if args.deepgate2_num_rounds < 1:
        raise ValueError("deepgate2_num_rounds must be positive")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.log_every_n_steps < 1:
        raise ValueError("log_every_n_steps must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    tanh_control_parameters(args.grad_scale_m, args.grad_scale_delta)
    if not 0 <= args.singleton_similarity_center <= 1:
        raise ValueError("singleton_similarity_center must lie in [0, 1]")
    if not 0 < args.singleton_softplus_width < float("inf"):
        raise ValueError("singleton_softplus_width must be finite and positive")
    if args.stage2_temperature <= 0:
        raise ValueError("stage2_temperature must be positive")
    if args.stage2_margin < 0:
        raise ValueError("stage2_margin must be non-negative")
    if args.singleton_circle_gamma <= 0:
        raise ValueError("singleton_circle_gamma must be positive")
    if args.singleton_negative_weight < 0:
        raise ValueError("singleton_negative_weight must be non-negative")
    if args.singleton_pair_chunk_size < 1:
        raise ValueError("singleton_pair_chunk_size must be positive")
    if args.checkpoint_free_bytes < 0:
        raise ValueError("checkpoint_free_bytes must be non-negative")
    if args.resume_checkpoint is not None:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume_checkpoint}"
            )
        if args.checkpoint is not None:
            raise ValueError(
                "--resume_checkpoint cannot be combined with initialization "
                "checkpoints"
            )
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    print(f"[Info] Loading data from {args.data_dir}...")

    train_dataset, val_dataset, test_dataset = load_packed_datasets(
        args.data_dir, load_test=False
    )
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty")
    if val_dataset is None:
        raise ValueError("Training requires a non-empty validation split")
    if train_dataset.feature_config != val_dataset.feature_config:
        raise ValueError("Train and validation feature_config differ")
    if (
        test_dataset is not None
        and train_dataset.feature_config != test_dataset.feature_config
    ):
        raise ValueError("Train and test feature_config differ")
    args.feature_config = dict(train_dataset.feature_config)
    resumed_epoch = (
        validate_resume_checkpoint(args.resume_checkpoint, args)
        if args.resume_checkpoint is not None
        else None
    )

    val_count = len(val_dataset)
    test_count = len(test_dataset) if test_dataset is not None else 0
    print(
        f"[Info] Loaded packed splits | Train: {len(train_dataset)} | "
        f"Val: {val_count} | Test: {test_count}"
    )

    train_loader = make_loader(train_dataset, args)
    val_loader = make_loader(val_dataset, args)
    print(
        f"[Info] Data loaders | batch_size={args.batch_size} | devices={args.devices} | "
        f"train_batches={len(train_loader)} | val_batches={len(val_loader)}"
    )

    model_detail = f"rounds={args.deepgate2_num_rounds}, pi_init=const"
    print(
        f"[Info] Initializing {args.encoder_type} "
        f"(hidden={args.hidden}, {model_detail})"
    )
    model = SECLModule(args)
    if args.deepgate2_gate_replay:
        if args.deepgate2_execution != "dag":
            raise ValueError("Gate replay requires DeepGate2 DAG execution")
        from model.deepgate2_gate_replay import install

        if args.deepgate2_replay_max_gates < 1:
            raise ValueError("deepgate2_replay_max_gates must be positive")
        model.gate_replay_cache = install(model.encoder, max_bucket=args.deepgate2_replay_max_gates)
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        initialize_encoder(model, args.checkpoint)
        print(f"[Info] Loaded encoder checkpoint: {args.checkpoint}")

    checkpoint_dir = args.checkpoint_dir or os.path.join(args.log_dir, "checkpoints")
    monitor, monitor_mode = TRAINING_MONITOR
    monitor_token = "{" + monitor + ":.6f}"
    checkpoint_callback = ResumableModelCheckpoint(
        monitor=monitor,
        dirpath=checkpoint_dir,
        filename=f"{{epoch:03d}}-{monitor_token}",
        save_top_k=1,
        save_last=True,
        mode=monitor_mode,
    )
    callbacks = [checkpoint_callback, EpochWallTimer()]
    if args.early_stopping_patience:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor=monitor, min_delta=0.0, patience=args.early_stopping_patience,
            verbose=True, mode=monitor_mode,
        ))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        default_root_dir=args.log_dir,
        logger=[
            TensorBoardLogger(save_dir=args.log_dir, name="lightning_logs"),
            CSVLogger(save_dir=args.log_dir, name="csv_logs"),
        ],
        callbacks=callbacks,
        enable_progress_bar=not args.no_progress_bar,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
    )

    if resumed_epoch is not None:
        print(
            "[Info] Resuming full training state after "
            f"epoch {resumed_epoch}: {args.resume_checkpoint}"
        )
    print("[Info] Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_checkpoint,
    )

    best_path = checkpoint_callback.best_model_path
    if best_path:
        restore_best_model(model, best_path)
        checkpoint_stem = args.encoder_type
        # Export the encoder alone; this is what abc/gnn/infer.py loads.
        encoder_path = os.path.join(checkpoint_dir, f"{checkpoint_stem}.ckpt")
        if trainer.is_global_zero:
            torch.save(model.encoder.checkpoint_payload(), encoder_path)
            print(f"[Info] Best Lightning checkpoint: {best_path}")
            print(f"[Info] Best encoder checkpoint: {encoder_path}")
    print("[Info] Training finished.")


if __name__ == "__main__":
    main()
