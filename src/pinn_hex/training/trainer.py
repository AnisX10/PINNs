from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from pinn_hex.models.pinn import DoublePipePINN
from pinn_hex.physics.double_pipe import Geometry, OperatingPoint
from pinn_hex.training.losses import compute_losses
from pinn_hex.utils.repro import resolve_device, set_seed


@dataclass
class TrainerArtifacts:
    history: list[dict[str, float]]
    best_loss: float
    checkpoint_path: Path
    predictions_path: Path
    metrics_path: Path


class PINNTrainer:
    def __init__(
        self,
        config: dict,
        geometry: Geometry,
        operating: OperatingPoint,
        hot_profile: pd.DataFrame,
        cold_profile: pd.DataFrame,
        mesh_classified: pd.DataFrame,
        hot_train_profile: pd.DataFrame | None = None,
        cold_train_profile: pd.DataFrame | None = None,
        hot_val_profile: pd.DataFrame | None = None,
        cold_val_profile: pd.DataFrame | None = None,
    ) -> None:
        self.config = config
        self.geometry = geometry
        self.operating = operating
        self.hot_profile = hot_profile.reset_index(drop=True)
        self.cold_profile = cold_profile.reset_index(drop=True)
        self.hot_train_profile = (
            hot_train_profile.reset_index(drop=True) if hot_train_profile is not None else self.hot_profile
        )
        self.cold_train_profile = (
            cold_train_profile.reset_index(drop=True) if cold_train_profile is not None else self.cold_profile
        )
        self.hot_val_profile = hot_val_profile.reset_index(drop=True) if hot_val_profile is not None else None
        self.cold_val_profile = cold_val_profile.reset_index(drop=True) if cold_val_profile is not None else None
        self.mesh = mesh_classified
        training_cfg = config["training"]
        model_cfg = config["model"]
        set_seed(int(training_cfg["seed"]))
        self.device = resolve_device(str(training_cfg["device"]))
        self.model = DoublePipePINN(
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_hidden_layers=int(model_cfg["num_hidden_layers"]),
            activation=str(model_cfg["activation"]),
            fourier_features=int(model_cfg["fourier_features"]),
            fourier_sigma=float(model_cfg["fourier_sigma"]),
            initial_u=float(model_cfg["initial_u_w_m2k"]),
            initial_alpha_hot=float(model_cfg["initial_alpha_hot_m2_s"]),
            initial_alpha_cold=float(model_cfg["initial_alpha_cold_m2_s"]),
            initial_hot_ambient_coupling=float(model_cfg["initial_hot_ambient_coupling_s_inv"]),
            hot_half_length=float(geometry.hot_half_length_m),
            cold_half_length=float(geometry.cold_half_length_m),
            time_scale_s=float(model_cfg["time_scale_s"]),
            hot_inlet_temperature_K=float(self.operating.hot_inlet_temperature_K),
            cold_inlet_temperature_K=float(self.operating.cold_inlet_temperature_K),
            temperature_scale_K=float(
                abs(self.operating.hot_inlet_temperature_K - self.operating.cold_inlet_temperature_K)
            ),
            cold_inlet_learnable=bool(model_cfg.get("cold_inlet_learnable", True)),
            cold_inlet_max_offset_fraction=float(model_cfg.get("cold_inlet_max_offset_fraction", 0.5)),
        ).to(self.device)

    def _tensor(self, values: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device, requires_grad=requires_grad).reshape(-1, 1)

    def _sample_z(self, region: str, n: int) -> np.ndarray:
        if region == "hot":
            return np.random.uniform(-self.geometry.hot_half_length_m, self.geometry.hot_half_length_m, size=n)
        if region == "cold":
            return np.random.uniform(-self.geometry.cold_half_length_m, self.geometry.cold_half_length_m, size=n)
        raise ValueError(f"Unsupported region for collocation: {region!r}.")

    def make_batch(self) -> dict[str, torch.Tensor]:
        training_cfg = self.config["training"]
        hot_z = self.hot_train_profile[self._z_column(self.hot_train_profile)].to_numpy()
        cold_z = self.cold_train_profile[self._z_column(self.cold_train_profile)].to_numpy()
        hot_t = np.zeros_like(hot_z)
        cold_t = np.zeros_like(cold_z)
        batch = {
            "z_hot_data": self._tensor(hot_z),
            "t_hot_data": self._tensor(hot_t),
            "T_hot_data": self._tensor(self.hot_train_profile[self._temperature_column(self.hot_train_profile)].to_numpy()),
            "W_hot_data": self._tensor(self._data_weights(self.hot_train_profile)),
            "z_cold_data": self._tensor(cold_z),
            "t_cold_data": self._tensor(cold_t),
            "T_cold_data": self._tensor(self.cold_train_profile[self._temperature_column(self.cold_train_profile)].to_numpy()),
            "W_cold_data": self._tensor(self._data_weights(self.cold_train_profile)),
            "z_hot_collocation": self._tensor(
                self._sample_z("hot", int(training_cfg["collocation_hot"])), requires_grad=True
            ),
            "t_hot_collocation": self._tensor(np.zeros(int(training_cfg["collocation_hot"])), requires_grad=True),
            "z_cold_collocation": self._tensor(
                self._sample_z("cold", int(training_cfg["collocation_cold"])), requires_grad=True
            ),
            "t_cold_collocation": self._tensor(np.zeros(int(training_cfg["collocation_cold"])), requires_grad=True),
            "z_hot_inlet": self._tensor(np.array([self.geometry.hot_half_length_m]), requires_grad=True),
            "t_hot_inlet": self._tensor(np.array([0.0]), requires_grad=True),
            "T_hot_inlet": self._tensor(np.array([self.operating.hot_inlet_temperature_K])),
            "z_cold_inlet": self._tensor(np.array([-self.geometry.cold_half_length_m]), requires_grad=True),
            "t_cold_inlet": self._tensor(np.array([0.0]), requires_grad=True),
            "T_cold_inlet": self._tensor(np.array([self.operating.cold_inlet_temperature_K])),
            "z_hot_outlet": self._tensor(np.array([-self.geometry.hot_half_length_m]), requires_grad=True),
            "t_hot_outlet": self._tensor(np.array([0.0]), requires_grad=True),
            "z_cold_outlet": self._tensor(np.array([self.geometry.cold_half_length_m]), requires_grad=True),
            "t_cold_outlet": self._tensor(np.array([0.0]), requires_grad=True),
            "z_hot_ic": self._tensor(
                np.linspace(-self.geometry.hot_half_length_m, self.geometry.hot_half_length_m, 64)
            ),
            "t_hot_ic": self._tensor(np.zeros(64)),
            "T_hot_ic": self._tensor(np.full(64, self.operating.initial_temperature_K)),
            "z_cold_ic": self._tensor(
                np.linspace(-self.geometry.cold_half_length_m, self.geometry.cold_half_length_m, 64)
            ),
            "t_cold_ic": self._tensor(np.zeros(64)),
            "T_cold_ic": self._tensor(np.full(64, self.operating.initial_temperature_K)),
        }
        return batch

    def fit(self, output_dir: str | Path, adam_epochs_override: int | None = None, skip_lbfgs: bool = False) -> TrainerArtifacts:
        output_path = Path(output_dir)
        (output_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_path / "figures").mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []
        training_cfg = self.config["training"]
        weights = {
            "data_weight": float(training_cfg["data_weight"]),
            "pde_weight": float(training_cfg["pde_weight"]),
            "bc_weight": float(training_cfg["bc_weight"]),
            "ic_weight": float(training_cfg["ic_weight"]),
            "reg_weight": float(training_cfg["reg_weight"]),
            "monotonic_weight": float(training_cfg.get("monotonic_weight", 0.0)),
            "ordering_weight": float(training_cfg.get("ordering_weight", 0.0)),
            "smoothness_weight": float(training_cfg.get("smoothness_weight", 0.0)),
            "cold_inlet_prior_weight": float(training_cfg.get("cold_inlet_prior_weight", 0.0)),
        }
        priors = {
            "initial_u": float(self.config["model"]["initial_u_w_m2k"]),
            "initial_alpha_hot": float(self.config["model"]["initial_alpha_hot_m2_s"]),
            "initial_alpha_cold": float(self.config["model"]["initial_alpha_cold_m2_s"]),
            "initial_hot_ambient_coupling": float(self.config["model"]["initial_hot_ambient_coupling_s_inv"]),
        }
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(training_cfg["adam_lr"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=100,
            min_lr=1.0e-5,
        )
        adam_epochs = int(adam_epochs_override or training_cfg["adam_epochs"])
        best_score = float("inf")
        checkpoint_path = output_path / "checkpoints" / "best_model.pt"
        for epoch in range(1, adam_epochs + 1):
            batch = self.make_batch()
            optimizer.zero_grad()
            losses = compute_losses(
                self.model,
                self.geometry,
                self.operating,
                batch,
                weights=self._phase_weights(epoch, adam_epochs, weights),
                priors=priors,
                stationary=bool(self.config["project"].get("mode", "stationary") == "stationary"),
            )
            losses["total"].backward()
            clip_grad_norm_(self.model.parameters(), max_norm=float(training_cfg["grad_clip"]))
            optimizer.step()
            record = {name: float(value.detach().cpu()) for name, value in losses.items()}
            record["epoch"] = epoch
            record["U"] = float(self.model.U.detach().cpu())
            record["cold_inlet_effective_K"] = float(self.model.cold_inlet_effective.detach().cpu())
            record["hot_ambient_coupling_s_inv"] = float(self.model.hot_ambient_coupling.detach().cpu())
            history.append(record)
            validation_metrics = self._compute_profile_metrics(self.hot_val_profile, self.cold_val_profile)
            validation_score = self._select_checkpoint_score(validation_metrics)
            scheduler.step(validation_score)
            record["validation_score"] = validation_score
            if validation_score < best_score:
                best_score = validation_score
                torch.save(self.model.state_dict(), checkpoint_path)
            if epoch == 1 or epoch % int(training_cfg["print_every"]) == 0 or epoch == adam_epochs:
                print(
                    f"epoch={epoch:05d} total={record['total']:.6f} "
                    f"data={record['data']:.6f} pde={record['pde']:.6f} "
                    f"val={validation_score:.4f} U={record['U']:.3f} Tc_in={record['cold_inlet_effective_K']:.3f}"
                )
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        if not skip_lbfgs and int(training_cfg["lbfgs_max_iter"]) > 0:
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=int(training_cfg["lbfgs_max_iter"]),
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                batch = self.make_batch()
                losses = compute_losses(
                    self.model,
                    self.geometry,
                    self.operating,
                    batch,
                    weights=weights,
                    priors=priors,
                    stationary=bool(self.config["project"].get("mode", "stationary") == "stationary"),
                )
                losses["total"].backward()
                return losses["total"]

            lbfgs.step(closure)
            lbfgs_validation = self._compute_profile_metrics(self.hot_val_profile, self.cold_val_profile)
            lbfgs_score = self._select_checkpoint_score(lbfgs_validation)
            if lbfgs_score < best_score:
                best_score = lbfgs_score
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        predictions = self.predict_profiles()
        predictions_path = output_path / "pinn_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        metrics_path = output_path / "training_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_validation_score": best_score,
                    "final_u_w_m2k": float(self.model.U.detach().cpu()),
                    "final_alpha_hot_m2_s": float(self.model.alpha_hot.detach().cpu()),
                    "final_alpha_cold_m2_s": float(self.model.alpha_cold.detach().cpu()),
                    "final_hot_ambient_coupling_s_inv": float(self.model.hot_ambient_coupling.detach().cpu()),
                    "final_cold_inlet_effective_K": float(self.model.cold_inlet_effective.detach().cpu()),
                    "epochs": len(history),
                    "train_metrics": self._compute_profile_metrics(self.hot_train_profile, self.cold_train_profile),
                    "validation_metrics": self._compute_profile_metrics(self.hot_val_profile, self.cold_val_profile),
                },
                handle,
                indent=2,
            )
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(output_path / "training_history.csv", index=False)
        self._save_plots(predictions, history_frame, output_path / "figures")
        return TrainerArtifacts(
            history=history,
            best_loss=best_score,
            checkpoint_path=checkpoint_path,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
        )

    def predict_profiles(self) -> pd.DataFrame:
        self.model.eval()
        with torch.no_grad():
            hot_z = self._tensor(self.hot_profile[self._z_column(self.hot_profile)].to_numpy())
            cold_z = self._tensor(self.cold_profile[self._z_column(self.cold_profile)].to_numpy())
            hot_t = self._tensor(np.zeros(len(self.hot_profile)))
            cold_t = self._tensor(np.zeros(len(self.cold_profile)))
            hot_pred = self.model(hot_z, hot_t).hot.detach().cpu().numpy().reshape(-1)
            cold_pred = self.model(cold_z, cold_t).cold.detach().cpu().numpy().reshape(-1)
        hot_frame = pd.DataFrame(
            {
                "stream": "hot",
                "z": self.hot_profile[self._z_column(self.hot_profile)],
                "s_flow": self.hot_profile["s_flow"],
                "T_data": self.hot_profile[self._temperature_column(self.hot_profile)],
                "T_pred": hot_pred,
            }
        )
        cold_frame = pd.DataFrame(
            {
                "stream": "cold",
                "z": self.cold_profile[self._z_column(self.cold_profile)],
                "s_flow": self.cold_profile["s_flow"],
                "T_data": self.cold_profile[self._temperature_column(self.cold_profile)],
                "T_pred": cold_pred,
            }
        )
        return pd.concat([hot_frame, cold_frame], ignore_index=True)

    @staticmethod
    def _temperature_column(frame: pd.DataFrame) -> str:
        if "T_supervision_K" in frame.columns:
            return "T_supervision_K"
        return "T_processed_K" if "T_processed_K" in frame.columns else "T_mean"

    @staticmethod
    def _data_weights(frame: pd.DataFrame) -> np.ndarray:
        if "T_std_interp_K" not in frame.columns:
            return np.ones(len(frame), dtype=np.float32)
        std = frame["T_std_interp_K"].to_numpy(dtype=np.float64)
        count = frame["n_interp"].to_numpy(dtype=np.float64) if "n_interp" in frame.columns else np.ones_like(std)
        std_ref = float(np.nanmedian(std[std > 0.0])) if np.any(std > 0.0) else 1.0
        raw = count / np.maximum(1.0, 1.0 + std / max(std_ref, 1.0e-6))
        raw = raw / np.maximum(np.mean(raw), 1.0e-6)
        return raw.astype(np.float32)

    @staticmethod
    def _z_column(frame: pd.DataFrame) -> str:
        return "z" if "z" in frame.columns else "z_mean"

    def _compute_profile_metrics(
        self,
        hot_frame: pd.DataFrame | None,
        cold_frame: pd.DataFrame | None,
    ) -> dict | None:
        if hot_frame is None or cold_frame is None:
            return None
        metrics: dict[str, dict[str, float]] = {}
        for stream, frame in [("hot", hot_frame), ("cold", cold_frame)]:
            if frame.empty:
                metrics[stream] = {"rmse_K": float("nan"), "mae_K": float("nan")}
                continue
            z = self._tensor(frame[self._z_column(frame)].to_numpy())
            t = self._tensor(np.zeros(len(frame)))
            with torch.no_grad():
                outputs = self.model(z, t)
                pred = outputs.hot if stream == "hot" else outputs.cold
            target = frame[self._temperature_column(frame)].to_numpy()
            pred_np = pred.detach().cpu().numpy().reshape(-1)
            rmse = float(np.sqrt(np.mean((pred_np - target) ** 2)))
            mae = float(np.mean(np.abs(pred_np - target)))
            metrics[stream] = {"rmse_K": rmse, "mae_K": mae}
        return metrics

    def _select_checkpoint_score(self, metrics: dict | None) -> float:
        if metrics is None:
            return float("inf")
        return float(metrics["hot"]["rmse_K"] + metrics["cold"]["rmse_K"])

    def _phase_weights(self, epoch: int, total_epochs: int, base: dict[str, float]) -> dict[str, float]:
        weights = dict(base)
        warmup_epochs = int(self.config["training"].get("warmup_epochs", 0))
        if warmup_epochs > 0 and epoch <= min(warmup_epochs, total_epochs):
            weights["pde_weight"] *= float(self.config["training"].get("warmup_pde_scale", 1.0))
            weights["bc_weight"] *= float(self.config["training"].get("warmup_bc_scale", 1.0))
            weights["monotonic_weight"] *= float(self.config["training"].get("warmup_monotonic_scale", 1.0))
        return weights

    def _save_plots(self, predictions: pd.DataFrame, history: pd.DataFrame, figures_dir: Path) -> None:
        plt.figure(figsize=(10, 5))
        for stream, color in [("hot", "tab:red"), ("cold", "tab:blue")]:
            group = predictions[predictions["stream"] == stream].sort_values("z")
            plt.plot(group["z"], group["T_data"], color=color, linewidth=2, label=f"{stream} data")
            plt.plot(group["z"], group["T_pred"], color=color, linestyle="--", linewidth=2, label=f"{stream} PINN")
        plt.xlabel("Axial coordinate z [m]")
        plt.ylabel("Temperature [K]")
        plt.title("PINN fit against reduced-order supervision profiles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "pinn_fit.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        for column in ["total", "data", "pde", "bc", "ic", "monotonic", "ordering", "smoothness"]:
            if column not in history.columns:
                continue
            plt.plot(history["epoch"], history[column], label=column)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PINN training losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "training_loss.png", dpi=200)
        plt.close()
