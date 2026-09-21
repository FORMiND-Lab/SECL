from __future__ import annotations

import pytorch_lightning as pl
import torch

from typing import NamedTuple
from .DeepGate2 import DeepGate2
from .deepgate2_schedule import prepare_cpu_schedule
from .loss import build_training_loss, loss_hyperparameters, validate_loss_checkpoint
from utils.structural_pe import build_feature_config


TRAINING_MONITOR = ("val_loss", "min")


class EncoderOutput(NamedTuple):
    hs: torch.Tensor
    hf: torch.Tensor


class SECLModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if getattr(args, "encoder_type", "DeepGate2") != "DeepGate2":
            raise ValueError("SECL uses the PI-aware DeepGate2 encoder")
        if getattr(args, "deepgate2_pi_init_mode", "const") != "const":
            raise ValueError("The paper initializes PI functional states to zero")
        self.save_hyperparameters(loss_hyperparameters(args))
        self.args = args
        self.feature_config = dict(getattr(args, "feature_config", build_feature_config()))
        self.encoder = DeepGate2(
            num_rounds=int(getattr(args, "deepgate2_num_rounds", 1)),
            dim_hidden=args.hidden,
            pi_init_mode="const",
        )

        self.encoder.readout_prob.requires_grad_(False)
        self._deepgate2_schedule_cache = {}
        self.loss = build_training_loss(args)

    def on_load_checkpoint(self, checkpoint):
        validate_loss_checkpoint(checkpoint.get("hyper_parameters", {}), self.hparams)

    def forward(
        self,
        batch,
        *,
        schedule_key=None,
        use_compact_schedule=True,
    ) -> EncoderOutput:
        compact_schedule = getattr(batch, "_deepgate2_compact_schedule", None)
        if use_compact_schedule and compact_schedule is None and schedule_key is not None:
            compact_schedule = self._deepgate2_schedule_cache.get(
                schedule_key
            )
            if compact_schedule is None:
                compact_schedule = self.encoder._compact_fanin_schedule(
                    batch.gate.reshape(-1),
                    batch.edge_index,
                    batch.forward_level.reshape(-1),
                )
                self._deepgate2_schedule_cache[
                    schedule_key
                ] = compact_schedule
        hs, hf = self.encoder(
            batch.gate,
            batch.edge_index,
            batch.forward_level,
            batch.forward_index,
            is_hs=True,
            batch=getattr(batch, "batch", None),
            _use_compact_schedule=use_compact_schedule,
            _compact_schedule=compact_schedule,
            _use_dag_state=getattr(self.args, "deepgate2_execution", "dag") != "dense",
            _use_gru_cell=getattr(self.args, "deepgate2_execution", "dag") == "dag",
        )
        return EncoderOutput(hs=hs, hf=hf)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        plan = getattr(batch, "_deepgate2_cpu_schedule", None)
        if plan is None and batch.gate.device.type == "cpu":
            plan = prepare_cpu_schedule(batch.gate, batch.edge_index, batch.forward_level)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if plan is not None:
            batch._deepgate2_compact_schedule = plan.to(device)
        return batch

    def _loss_components(
        self,
        output: EncoderOutput,
        batch,
        *,
        update_balance: bool | None = None,
        compute_diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        equivalence = self.loss(
            output.hf,
            batch.y_index,
            batch.y_sign,
            batch.sim_class,
            batch.sim_class_sign,
            batch.batch,
            batch.gate,
            y_known=batch.y_known,
            collect_diagnostics=compute_diagnostics,
        )

        def scalar(value):
            return torch.tensor(value, device=output.hf.device, dtype=torch.float64)
        return {
            "stage2_main_loss": equivalence.loss,
            "prototype_secl_loss": equivalence.secl_loss,
            "singleton_negative_loss": equivalence.singleton_negative_loss,
            "stage2_active_graphs": scalar(equivalence.active_graphs),
            "stage2_singleton_negative_graphs": scalar(
                equivalence.singleton_negative_graphs
            ),
            "stage2_simulation_classes": scalar(
                equivalence.simulation_classes
            ),
            "stage2_singleton_classes": scalar(
                equivalence.singleton_classes
            ),
            "stage2_candidate_nodes": scalar(equivalence.candidate_nodes),
            "stage2_anchors": scalar(equivalence.anchors),
            "stage2_entities": scalar(equivalence.entities),
            "stage2_prototypes": scalar(equivalence.prototypes),
            "stage2_singleton_keys": scalar(equivalence.singleton_keys),
            "stage2_relations": scalar(equivalence.relations),
            "stage2_singleton_relations": scalar(
                equivalence.singleton_relations
            ),
            "stage2_singleton_pair_relations": scalar(
                equivalence.singleton_pair_relations
            ),
            "stage2_singleton_violation_relations": scalar(
                equivalence.singleton_violation_relations
            ),
            "stage2_prototype_fallbacks": scalar(
                equivalence.prototype_fallbacks
            ),
            "stage2_leave_one_out_fallbacks": scalar(
                equivalence.leave_one_out_fallbacks
            ),
            "stage2_streamed_graphs": scalar(equivalence.streamed_graphs),
            "stage2_streamed_singleton_classes": scalar(
                equivalence.streamed_singleton_classes
            ),
            "total": equivalence.loss,
        }

    def on_validation_epoch_start(self) -> None:
        self._deepgate2_schedule_cache.clear()

    def on_train_epoch_start(self) -> None:
        self._deepgate2_schedule_cache.clear()

    def _shared_step(
        self, batch, prefix: str, update_balance: bool, batch_idx: int
    ):
        output = self.forward(
            batch, schedule_key=(prefix, int(batch_idx))
        )
        components = self._loss_components(
            output,
            batch,
            update_balance=update_balance,
            compute_diagnostics=False,
        )
        active_graphs = int(components["stage2_active_graphs"].detach().item())
        log_args = {
            "on_step": prefix == "train",
            "on_epoch": True,
            "logger": True,

            "batch_size": active_graphs,
        }
        if active_graphs > 0:
            self.log(
                f"{prefix}_loss",
                components["total"],
                prog_bar=True,
                **log_args,
            )
            for name in ("prototype_secl_loss", "singleton_negative_loss"):
                self.log(f"{prefix}_{name}", components[name], **log_args)
        return components["total"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", True, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val", False, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            lr=self.args.lr,
            weight_decay=1e-5,
        )
        monitor, mode = TRAINING_MONITOR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=0.5, patience=20, min_lr=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }
