from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from pinn_hex.data.case_matrix import OPERATING_COLUMNS
from pinn_hex.models.factory_3d import build_double_pipe_pinn_3d
from pinn_hex.physics.double_pipe import OperatingPoint
from pinn_hex.physics.double_pipe_3d import (
    ThreeDGeometry,
    sample_annulus_disk,
    sample_annulus_volume,
    sample_cylinder_volume,
    sample_disk,
    sample_interface_pair,
)
from pinn_hex.physics.multiphysics_3d import conjugate_interface_residuals, fluid_residuals, wall_residuals
from pinn_hex.utils.repro import resolve_device, set_seed


@dataclass
class Trainer3DArtifacts:
    history: list[dict[str, float]]
    best_score: float
    checkpoint_path: Path
    predictions_path: Path
    metrics_path: Path


class PINNTrainer3D:
    def __init__(
        self,
        config: dict,
        geometry: ThreeDGeometry,
        operating: OperatingPoint,
        hot_train: pd.DataFrame,
        cold_train: pd.DataFrame,
        hot_validation: pd.DataFrame,
        cold_validation: pd.DataFrame,
    ) -> None:
        self.config = config
        self.geometry = geometry
        self.operating = operating
        self.hot_train = hot_train.reset_index(drop=True)
        self.cold_train = cold_train.reset_index(drop=True)
        self.hot_validation = hot_validation.reset_index(drop=True)
        self.cold_validation = cold_validation.reset_index(drop=True)
        self.operating_columns = list(OPERATING_COLUMNS)

        training_cfg = config["training_3d"]
        model_cfg = config["model_3d"]
        set_seed(int(training_cfg["seed"]))
        self.rng = np.random.default_rng(int(training_cfg["seed"]))
        self.device = resolve_device(str(training_cfg["device"]))
        self.condition_on_operating_point = bool(model_cfg.get("condition_on_operating_point", False))
        self.hot_inlet_target_K = self._resolve_temperature_target(
            self.hot_train,
            boundary="hot_inlet",
            source=str(training_cfg.get("hot_inlet_temperature_source", "reference")),
            fallback=float(operating.hot_inlet_temperature_K),
        )
        self.cold_inlet_target_K = self._resolve_temperature_target(
            self.cold_train,
            boundary="cold_inlet",
            source=str(training_cfg.get("cold_inlet_temperature_source", "train_boundary_mean")),
            fallback=float(operating.cold_inlet_temperature_K),
        )
        self.surface_sampling_weights = {
            str(key): float(value)
            for key, value in training_cfg.get("surface_boundary_sampling_weights", {}).items()
        }
        self.surface_loss_weights = {
            str(key): float(value)
            for key, value in training_cfg.get("surface_boundary_loss_weights", {}).items()
        }
        self.training_case_bank = self._build_training_case_bank()
        self.case_aggregate_targets = self._build_case_aggregate_targets_for_frames(self.hot_train, self.cold_train)
        self.validation_case_aggregate_targets = self._build_case_aggregate_targets_for_frames(
            self.hot_validation,
            self.cold_validation,
        )
        self.model = build_double_pipe_pinn_3d(
            config=config,
            geometry=geometry,
            operating=operating,
            hot_inlet_temperature_K=float(self.hot_inlet_target_K),
            cold_inlet_temperature_K=float(self.cold_inlet_target_K),
        ).to(self.device)

    def _tensor(self, values: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device, requires_grad=requires_grad)

    def _zero_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    @staticmethod
    def _xyz_from_frame(frame: pd.DataFrame) -> np.ndarray:
        return frame[["x", "y", "z"]].to_numpy(dtype=np.float32)

    @staticmethod
    def _column_from_frame(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
        if column in frame.columns:
            return frame[column].to_numpy(dtype=np.float32)
        return np.full(len(frame), default, dtype=np.float32)

    def _ops_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if all(column in frame.columns for column in self.operating_columns):
            return frame[self.operating_columns].to_numpy(dtype=np.float32)
        default = np.array(
            [
                float(self.operating.hot_inlet_temperature_K),
                float(self.operating.cold_inlet_temperature_K),
                float(self.operating.inlet_velocity_hot_m_per_s),
                float(self.operating.inlet_velocity_cold_m_per_s),
            ],
            dtype=np.float32,
        )
        return np.repeat(default.reshape(1, -1), len(frame), axis=0)

    def _build_training_case_bank(self) -> np.ndarray:
        combined = pd.concat([self.hot_train, self.cold_train], ignore_index=True)
        if not all(column in combined.columns for column in self.operating_columns):
            default = np.array(
                [
                    float(self.operating.hot_inlet_temperature_K),
                    float(self.operating.cold_inlet_temperature_K),
                    float(self.operating.inlet_velocity_hot_m_per_s),
                    float(self.operating.inlet_velocity_cold_m_per_s),
                ],
                dtype=np.float32,
            )
            return default.reshape(1, -1)
        dedupe_columns = [column for column in ["case_id", *self.operating_columns] if column in combined.columns]
        bank = combined[dedupe_columns].drop_duplicates().reset_index(drop=True)
        return bank[self.operating_columns].to_numpy(dtype=np.float32)

    def _sample_operating_points(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, len(self.operating_columns)), dtype=np.float32)
        if len(self.training_case_bank) == 1:
            return np.repeat(self.training_case_bank, n, axis=0)
        indices = self.rng.choice(len(self.training_case_bank), size=n, replace=True)
        return self.training_case_bank[indices]

    def _mean_training_operating_point(self) -> np.ndarray:
        if len(self.training_case_bank) == 0:
            return np.array(
                [
                    float(self.operating.hot_inlet_temperature_K),
                    float(self.operating.cold_inlet_temperature_K),
                    float(self.operating.inlet_velocity_hot_m_per_s),
                    float(self.operating.inlet_velocity_cold_m_per_s),
                ],
                dtype=np.float32,
            )
        return self.training_case_bank.mean(axis=0)

    @staticmethod
    def _weighted_mean_numpy(values: np.ndarray, weights: np.ndarray | None = None) -> float:
        values64 = values.astype(np.float64)
        if weights is None:
            return float(values64.mean())
        weights64 = np.abs(weights.astype(np.float64))
        total_weight = float(weights64.sum())
        if total_weight <= 1.0e-12:
            return float(values64.mean())
        return float(np.sum(values64 * weights64) / total_weight)

    @staticmethod
    def _weighted_mean_tensor(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights_abs = torch.abs(weights)
        total_weight = torch.clamp(torch.sum(weights_abs), min=1.0e-8)
        return torch.sum(values * weights_abs) / total_weight

    def _build_case_aggregate_targets_for_frames(
        self,
        hot_frame: pd.DataFrame,
        cold_frame: pd.DataFrame,
    ) -> list[dict[str, object]]:
        if "case_id" not in hot_frame.columns or "case_id" not in cold_frame.columns:
            return []
        targets: list[dict[str, object]] = []
        hot_case_ids = [str(case_id) for case_id in hot_frame["case_id"].drop_duplicates().tolist()]
        cold_case_ids = {str(case_id) for case_id in cold_frame["case_id"].drop_duplicates().tolist()}
        for case_id in hot_case_ids:
            if case_id not in cold_case_ids:
                continue
            hot_case = hot_frame[hot_frame["case_id"] == case_id].reset_index(drop=True)
            cold_case = cold_frame[cold_frame["case_id"] == case_id].reset_index(drop=True)
            hot_inlet = hot_case[hot_case["boundary"] == "hot_inlet"].reset_index(drop=True)
            hot_outlet = hot_case[hot_case["boundary"] == "hot_outlet"].reset_index(drop=True)
            cold_inlet = cold_case[cold_case["boundary"] == "cold_inlet"].reset_index(drop=True)
            cold_outlet = cold_case[cold_case["boundary"] == "cold_outlet"].reset_index(drop=True)
            if hot_inlet.empty or hot_outlet.empty or cold_inlet.empty or cold_outlet.empty:
                continue
            targets.append(
                {
                    "case_id": case_id,
                    "hot_inlet_xyz": self._tensor(self._xyz_from_frame(hot_inlet)),
                    "hot_inlet_ops": self._tensor(self._ops_from_frame(hot_inlet)),
                    "hot_outlet_xyz": self._tensor(self._xyz_from_frame(hot_outlet)),
                    "hot_outlet_ops": self._tensor(self._ops_from_frame(hot_outlet)),
                    "cold_inlet_xyz": self._tensor(self._xyz_from_frame(cold_inlet)),
                    "cold_inlet_ops": self._tensor(self._ops_from_frame(cold_inlet)),
                    "cold_outlet_xyz": self._tensor(self._xyz_from_frame(cold_outlet)),
                    "cold_outlet_ops": self._tensor(self._ops_from_frame(cold_outlet)),
                    "hot_inlet_weight": self._tensor(np.abs(self._column_from_frame(hot_inlet, "w"))).reshape(-1, 1),
                    "hot_outlet_weight": self._tensor(np.abs(self._column_from_frame(hot_outlet, "w"))).reshape(-1, 1),
                    "cold_inlet_weight": self._tensor(np.abs(self._column_from_frame(cold_inlet, "w"))).reshape(-1, 1),
                    "cold_outlet_weight": self._tensor(np.abs(self._column_from_frame(cold_outlet, "w"))).reshape(-1, 1),
                    "hot_inlet_bulk_T": self._weighted_mean_numpy(
                        self._column_from_frame(hot_inlet, "T"),
                        self._column_from_frame(hot_inlet, "w"),
                    ),
                    "hot_outlet_bulk_T": self._weighted_mean_numpy(
                        self._column_from_frame(hot_outlet, "T"),
                        self._column_from_frame(hot_outlet, "w"),
                    ),
                    "cold_inlet_bulk_T": self._weighted_mean_numpy(
                        self._column_from_frame(cold_inlet, "T"),
                        self._column_from_frame(cold_inlet, "w"),
                    ),
                    "cold_outlet_bulk_T": self._weighted_mean_numpy(
                        self._column_from_frame(cold_outlet, "T"),
                        self._column_from_frame(cold_outlet, "w"),
                    ),
                    "hot_dp_Pa": float(
                        hot_case["dp_hot"].iloc[0]
                        if "dp_hot" in hot_case.columns
                        else self._column_from_frame(hot_inlet, "p").mean() - self._column_from_frame(hot_outlet, "p").mean()
                    ),
                    "cold_dp_Pa": float(
                        cold_case["dp_cold"].iloc[0]
                        if "dp_cold" in cold_case.columns
                        else self._column_from_frame(cold_inlet, "p").mean() - self._column_from_frame(cold_outlet, "p").mean()
                    ),
                    "hot_mass_flow_kg_s": float(
                        hot_case["m_dot_hot"].iloc[0]
                        if "m_dot_hot" in hot_case.columns
                        else self.operating.density_kg_per_m3
                        * self.geometry.hot_area_m2
                        * float(hot_inlet[self.operating_columns[2]].iloc[0])
                    ),
                    "cold_mass_flow_kg_s": float(
                        cold_case["m_dot_cold"].iloc[0]
                        if "m_dot_cold" in cold_case.columns
                        else self.operating.density_kg_per_m3
                        * self.geometry.cold_area_m2
                        * float(cold_inlet[self.operating_columns[3]].iloc[0])
                    ),
                }
            )
            if "Th_in_bulk" in hot_case.columns:
                targets[-1]["hot_inlet_bulk_T"] = float(hot_case["Th_in_bulk"].iloc[0])
            if "Th_out_bulk" in hot_case.columns:
                targets[-1]["hot_outlet_bulk_T"] = float(hot_case["Th_out_bulk"].iloc[0])
            if "Tc_in_bulk" in cold_case.columns:
                targets[-1]["cold_inlet_bulk_T"] = float(cold_case["Tc_in_bulk"].iloc[0])
            if "Tc_out_bulk" in cold_case.columns:
                targets[-1]["cold_outlet_bulk_T"] = float(cold_case["Tc_out_bulk"].iloc[0])
            targets[-1]["reference_q_total_w"] = float(
                hot_case["Q_total"].iloc[0]
                if "Q_total" in hot_case.columns
                else 0.5
                * (
                    targets[-1]["hot_mass_flow_kg_s"]
                    * float(self.operating.cp_J_per_kgK)
                    * (targets[-1]["hot_inlet_bulk_T"] - targets[-1]["hot_outlet_bulk_T"])
                    + targets[-1]["cold_mass_flow_kg_s"]
                    * float(self.operating.cp_J_per_kgK)
                    * (targets[-1]["cold_outlet_bulk_T"] - targets[-1]["cold_inlet_bulk_T"])
                )
            )
        return targets

    def _compute_wall_validation_metrics(self) -> dict[str, float]:
        hot_wall = self.hot_validation[self.hot_validation["boundary"] == "hot_wall"].reset_index(drop=True)
        cold_inner_wall = self.cold_validation[self.cold_validation["boundary"] == "cold_inner_wall"].reset_index(drop=True)
        metrics: dict[str, float] = {
            "hot_wall_target_rmse_K": 0.0,
            "cold_wall_target_rmse_K": 0.0,
            "hot_wall_stream_gap_rmse_K": 0.0,
            "cold_wall_stream_gap_rmse_K": 0.0,
        }
        if not hot_wall.empty:
            hot_xyz = self._tensor(self._xyz_from_frame(hot_wall))
            hot_ops = self._tensor(self._ops_from_frame(hot_wall))
            hot_target = self._tensor(self._column_from_frame(hot_wall, "T")).reshape(-1, 1)
            with torch.no_grad():
                hot_wall_state = self.model.wall(hot_xyz, hot_ops)
                hot_stream_state = self.model.hot(hot_xyz, hot_ops)
            metrics["hot_wall_target_rmse_K"] = float(
                torch.sqrt(torch.mean((hot_wall_state.T - hot_target) ** 2)).detach().cpu()
            )
            metrics["hot_wall_stream_gap_rmse_K"] = float(
                torch.sqrt(torch.mean((hot_wall_state.T - hot_stream_state.T) ** 2)).detach().cpu()
            )
        if not cold_inner_wall.empty:
            cold_xyz = self._tensor(self._xyz_from_frame(cold_inner_wall))
            cold_ops = self._tensor(self._ops_from_frame(cold_inner_wall))
            cold_target = self._tensor(self._column_from_frame(cold_inner_wall, "T")).reshape(-1, 1)
            with torch.no_grad():
                cold_wall_state = self.model.wall(cold_xyz, cold_ops)
                cold_stream_state = self.model.cold(cold_xyz, cold_ops)
            metrics["cold_wall_target_rmse_K"] = float(
                torch.sqrt(torch.mean((cold_wall_state.T - cold_target) ** 2)).detach().cpu()
            )
            metrics["cold_wall_stream_gap_rmse_K"] = float(
                torch.sqrt(torch.mean((cold_wall_state.T - cold_stream_state.T) ** 2)).detach().cpu()
            )
        return metrics

    def _compute_validation_objectives(self) -> dict[str, float]:
        validation = self._compute_validation_metrics()
        score = float(validation["hot"]["rmse_K"] + validation["cold"]["rmse_K"])
        summary = {
            "surface_rmse_K": score,
            "mean_q_rel_error_pct": 0.0,
            "mean_energy_gap_pct": 0.0,
            "mean_dp_rel_error_pct": 0.0,
            "mean_wall_target_rmse_K": 0.0,
            "mean_wall_stream_gap_rmse_K": 0.0,
        }
        if self.validation_case_aggregate_targets:
            q_errors: list[float] = []
            energy_gaps: list[float] = []
            dp_errors: list[float] = []
            for case_target in self.validation_case_aggregate_targets:
                with torch.no_grad():
                    hot_inlet_state_case = self.model.hot(case_target["hot_inlet_xyz"], case_target["hot_inlet_ops"])
                    hot_outlet_state_case = self.model.hot(case_target["hot_outlet_xyz"], case_target["hot_outlet_ops"])
                    cold_inlet_state_case = self.model.cold(case_target["cold_inlet_xyz"], case_target["cold_inlet_ops"])
                    cold_outlet_state_case = self.model.cold(case_target["cold_outlet_xyz"], case_target["cold_outlet_ops"])
                    hot_outlet_bulk_pred = float(
                        self._weighted_mean_tensor(
                            hot_outlet_state_case.T,
                            case_target["hot_outlet_weight"],
                        ).detach().cpu()
                    )
                    cold_outlet_bulk_pred = float(
                        self._weighted_mean_tensor(
                            cold_outlet_state_case.T,
                            case_target["cold_outlet_weight"],
                        ).detach().cpu()
                    )
                    hot_dp_pred = float((torch.mean(hot_inlet_state_case.p) - torch.mean(hot_outlet_state_case.p)).detach().cpu())
                    cold_dp_pred = float(
                        (torch.mean(cold_inlet_state_case.p) - torch.mean(cold_outlet_state_case.p)).detach().cpu()
                    )
                q_hot_pred = (
                    float(case_target["hot_mass_flow_kg_s"])
                    * float(self.operating.cp_J_per_kgK)
                    * (float(case_target["hot_inlet_bulk_T"]) - hot_outlet_bulk_pred)
                )
                q_cold_pred = (
                    float(case_target["cold_mass_flow_kg_s"])
                    * float(self.operating.cp_J_per_kgK)
                    * (cold_outlet_bulk_pred - float(case_target["cold_inlet_bulk_T"]))
                )
                q_mean_pred = 0.5 * (q_hot_pred + q_cold_pred)
                reference_q = float(case_target["reference_q_total_w"])
                q_scale = max(abs(reference_q), 1.0e-8)
                energy_scale = max(abs(reference_q), abs(q_hot_pred), abs(q_cold_pred), 1.0e-8)
                hot_dp_scale = max(abs(float(case_target["hot_dp_Pa"])), 1.0e-8)
                cold_dp_scale = max(abs(float(case_target["cold_dp_Pa"])), 1.0e-8)
                q_errors.append(abs(q_mean_pred - reference_q) / q_scale * 100.0)
                energy_gaps.append(abs(q_hot_pred - q_cold_pred) / energy_scale * 100.0)
                dp_errors.append(0.5 * (abs(hot_dp_pred - float(case_target["hot_dp_Pa"])) / hot_dp_scale * 100.0))
                dp_errors.append(0.5 * (abs(cold_dp_pred - float(case_target["cold_dp_Pa"])) / cold_dp_scale * 100.0))
            summary["mean_q_rel_error_pct"] = float(np.mean(q_errors))
            summary["mean_energy_gap_pct"] = float(np.mean(energy_gaps))
            summary["mean_dp_rel_error_pct"] = float(np.mean(dp_errors))
        wall_metrics = self._compute_wall_validation_metrics()
        summary["mean_wall_target_rmse_K"] = 0.5 * (
            wall_metrics["hot_wall_target_rmse_K"] + wall_metrics["cold_wall_target_rmse_K"]
        )
        summary["mean_wall_stream_gap_rmse_K"] = 0.5 * (
            wall_metrics["hot_wall_stream_gap_rmse_K"] + wall_metrics["cold_wall_stream_gap_rmse_K"]
        )
        temp_scale = float(self.config["model_3d"]["temperature_scale_K"])
        validation_cfg = self.config["training_3d"]
        score = (
            score
            + float(validation_cfg.get("validation_q_rel_weight", 0.0)) * summary["mean_q_rel_error_pct"] / 100.0
            + float(validation_cfg.get("validation_energy_gap_weight", 0.0))
            * summary["mean_energy_gap_pct"]
            / 100.0
            + float(validation_cfg.get("validation_dp_rel_weight", 0.0)) * summary["mean_dp_rel_error_pct"] / 100.0
            + float(validation_cfg.get("validation_wall_target_weight", 0.0))
            * summary["mean_wall_target_rmse_K"]
            / temp_scale
            + float(validation_cfg.get("validation_wall_stream_gap_weight", 0.0))
            * summary["mean_wall_stream_gap_rmse_K"]
            / temp_scale
        )
        summary["score"] = float(score)
        return summary

    def _resolved_inlet_temperature_summary(self) -> dict[str, float]:
        hot_bias = float(self.model.hot_inlet_temperature_bias.detach().cpu())
        cold_bias = float(self.model.cold_inlet_temperature_bias.detach().cpu())
        if not self.condition_on_operating_point:
            return {
                "hot_inlet_temperature_K": float(self.model.hot_inlet_temperature.detach().cpu()),
                "cold_inlet_temperature_K": float(self.model.cold_inlet_temperature.detach().cpu()),
                "hot_inlet_temperature_bias_K": hot_bias,
                "cold_inlet_temperature_bias_K": cold_bias,
            }
        mean_operating = self._mean_training_operating_point()
        return {
            "hot_inlet_temperature_K": float(mean_operating[0] + hot_bias),
            "cold_inlet_temperature_K": float(mean_operating[1] + cold_bias),
            "hot_inlet_temperature_bias_K": hot_bias,
            "cold_inlet_temperature_bias_K": cold_bias,
        }

    @staticmethod
    def _case_ids(frame: pd.DataFrame) -> list[str]:
        if "case_id" not in frame.columns:
            return []
        return [str(case_id) for case_id in frame["case_id"].drop_duplicates().tolist()]

    @staticmethod
    def _resolve_temperature_target(
        frame: pd.DataFrame,
        boundary: str,
        source: str,
        fallback: float,
    ) -> float:
        source_name = source.lower().strip()
        if source_name == "reference":
            return fallback
        if source_name == "train_boundary_mean":
            subset = frame[frame["boundary"] == boundary]
            if not subset.empty:
                return float(subset["T"].mean())
            return fallback
        raise ValueError(f"Unsupported temperature target source: {source}")

    def _boundary_weight_array(self, frame: pd.DataFrame, configured_weights: dict[str, float]) -> np.ndarray:
        if frame.empty:
            return np.empty((0,), dtype=np.float32)
        weights = np.ones(len(frame), dtype=np.float32)
        if not configured_weights:
            return weights
        for boundary, value in configured_weights.items():
            weights[frame["boundary"].to_numpy() == boundary] = max(float(value), 0.0)
        return weights

    @staticmethod
    def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        residual = (prediction - target) ** 2
        return torch.sum(weights * residual) / torch.clamp(torch.sum(weights), min=1.0e-8)

    def _surface_observation_tensors(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        return {
            "xyz": self._tensor(self._xyz_from_frame(frame)),
            "ops": self._tensor(self._ops_from_frame(frame)),
            "T": self._tensor(self._column_from_frame(frame, "T")).reshape(-1, 1),
            "u": self._tensor(self._column_from_frame(frame, "u")).reshape(-1, 1),
            "v": self._tensor(self._column_from_frame(frame, "v")).reshape(-1, 1),
            "w": self._tensor(self._column_from_frame(frame, "w")).reshape(-1, 1),
            "p": self._tensor(self._column_from_frame(frame, "p")).reshape(-1, 1),
            "weight": self._tensor(self._boundary_weight_array(frame, self.surface_loss_weights)).reshape(-1, 1),
        }

    def _wall_supervision_tensors(self, hot_frame: pd.DataFrame, cold_frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        hot_wall = hot_frame[hot_frame["boundary"] == "hot_wall"].reset_index(drop=True)
        cold_inner_wall = cold_frame[cold_frame["boundary"] == "cold_inner_wall"].reset_index(drop=True)
        return {
            "inner_xyz": self._tensor(self._xyz_from_frame(hot_wall), requires_grad=True),
            "inner_ops": self._tensor(self._ops_from_frame(hot_wall)),
            "inner_T": self._tensor(self._column_from_frame(hot_wall, "T")).reshape(-1, 1),
            "inner_qn": self._tensor(self._column_from_frame(hot_wall, "qn")).reshape(-1, 1),
            "inner_normal": self._tensor(
                np.column_stack(
                    [
                        self._column_from_frame(hot_wall, "nx"),
                        self._column_from_frame(hot_wall, "ny"),
                        self._column_from_frame(hot_wall, "nz"),
                    ]
                )
            ),
            "inner_weight": self._tensor(self._boundary_weight_array(hot_wall, self.surface_loss_weights)).reshape(-1, 1),
            "outer_xyz": self._tensor(self._xyz_from_frame(cold_inner_wall), requires_grad=True),
            "outer_ops": self._tensor(self._ops_from_frame(cold_inner_wall)),
            "outer_T": self._tensor(self._column_from_frame(cold_inner_wall, "T")).reshape(-1, 1),
            "outer_qn": self._tensor(self._column_from_frame(cold_inner_wall, "qn")).reshape(-1, 1),
            "outer_normal": self._tensor(
                np.column_stack(
                    [
                        self._column_from_frame(cold_inner_wall, "nx"),
                        self._column_from_frame(cold_inner_wall, "ny"),
                        self._column_from_frame(cold_inner_wall, "nz"),
                    ]
                )
            ),
            "outer_weight": self._tensor(
                self._boundary_weight_array(cold_inner_wall, self.surface_loss_weights)
            ).reshape(-1, 1),
        }

    def _sample_surface_batch(self, frame: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(frame) <= n:
            return frame
        balance_cases = bool(self.config["training_3d"].get("balance_cases_in_surface_batches", False))
        group_columns = ["boundary"]
        if balance_cases and "case_id" in frame.columns:
            group_columns = ["case_id", "boundary"]
        groups = list(frame.groupby(group_columns, sort=False))
        if not groups:
            return frame.iloc[:n].reset_index(drop=True)
        configured_weight_values: list[float] = []
        for name, _ in groups:
            if isinstance(name, tuple):
                boundary_name = str(name[-1])
            else:
                boundary_name = str(name)
            configured_weight_values.append(max(self.surface_sampling_weights.get(boundary_name, 1.0), 0.0))
        configured_weights = np.array(configured_weight_values, dtype=np.float64)
        if not np.any(configured_weights > 0.0):
            configured_weights = np.ones(len(groups), dtype=np.float64)
        exact_counts = n * configured_weights / configured_weights.sum()
        target_counts = np.floor(exact_counts).astype(int)
        remainder = n - int(target_counts.sum())
        if remainder > 0:
            order = np.argsort(-(exact_counts - target_counts))
            for idx in order[:remainder]:
                target_counts[idx] += 1
        selected_indices: list[int] = []
        for target_count, (_, group) in zip(target_counts, groups):
            if target_count <= 0:
                continue
            if len(group) <= target_count:
                selected_indices.extend(group.index.tolist())
                continue
            indices = self.rng.choice(len(group), size=target_count, replace=False)
            selected_indices.extend(group.iloc[np.sort(indices)].index.tolist())
        selected_indices = list(dict.fromkeys(selected_indices))
        if len(selected_indices) < n:
            remaining = frame.drop(index=selected_indices, errors="ignore")
            if not remaining.empty:
                extra_n = min(n - len(selected_indices), len(remaining))
                extra_idx = self.rng.choice(len(remaining), size=extra_n, replace=False)
                selected_indices.extend(remaining.iloc[np.sort(extra_idx)].index.tolist())
        return frame.loc[selected_indices[:n]].reset_index(drop=True)

    def make_batch(self) -> dict[str, torch.Tensor]:
        cfg = self.config["training_3d"]
        hot_surface = self._sample_surface_batch(self.hot_train, int(cfg["surface_batch_hot"]))
        cold_surface = self._sample_surface_batch(self.cold_train, int(cfg["surface_batch_cold"]))
        hot_surface_obs = self._surface_observation_tensors(hot_surface)
        cold_surface_obs = self._surface_observation_tensors(cold_surface)
        wall_surface_obs = self._wall_supervision_tensors(hot_surface, cold_surface)

        hot_collocation = sample_cylinder_volume(
            int(cfg["collocation_hot"]),
            self.geometry.hot_radius_m,
            -self.geometry.hot_half_length_m,
            self.geometry.hot_half_length_m,
            self.rng,
        )
        cold_collocation = sample_annulus_volume(
            int(cfg["collocation_cold"]),
            self.geometry.cold_inner_radius_m,
            self.geometry.cold_outer_radius_m,
            -self.geometry.cold_half_length_m,
            self.geometry.cold_half_length_m,
            self.rng,
        )
        wall_collocation = sample_annulus_volume(
            int(cfg["collocation_wall"]),
            self.geometry.hot_radius_m,
            self.geometry.cold_inner_radius_m,
            -self.geometry.cold_half_length_m,
            self.geometry.cold_half_length_m,
            self.rng,
        )
        hot_inlet = sample_disk(
            int(cfg["inlet_points_hot"]),
            self.geometry.hot_radius_m,
            self.geometry.hot_half_length_m,
            self.rng,
        )
        cold_inlet = sample_annulus_disk(
            int(cfg["inlet_points_cold"]),
            self.geometry.cold_inner_radius_m,
            self.geometry.cold_outer_radius_m,
            -self.geometry.cold_half_length_m,
            self.rng,
        )
        hot_outlet = sample_disk(
            int(cfg["outlet_points_hot"]),
            self.geometry.hot_radius_m,
            -self.geometry.hot_half_length_m,
            self.rng,
        )
        cold_outlet = sample_annulus_disk(
            int(cfg["outlet_points_cold"]),
            self.geometry.cold_inner_radius_m,
            self.geometry.cold_outer_radius_m,
            self.geometry.cold_half_length_m,
            self.rng,
        )
        hot_interface, cold_interface = sample_interface_pair(
            int(cfg["interface_points"]),
            self.geometry.hot_radius_m,
            self.geometry.cold_inner_radius_m,
            -self.geometry.cold_half_length_m,
            self.geometry.cold_half_length_m,
            self.rng,
        )
        return {
            "hot_surface_xyz": hot_surface_obs["xyz"],
            "hot_surface_ops": hot_surface_obs["ops"],
            "hot_surface_T": hot_surface_obs["T"],
            "hot_surface_u": hot_surface_obs["u"],
            "hot_surface_v": hot_surface_obs["v"],
            "hot_surface_w": hot_surface_obs["w"],
            "hot_surface_p": hot_surface_obs["p"],
            "hot_surface_weight": hot_surface_obs["weight"],
            "cold_surface_xyz": cold_surface_obs["xyz"],
            "cold_surface_ops": cold_surface_obs["ops"],
            "cold_surface_T": cold_surface_obs["T"],
            "cold_surface_u": cold_surface_obs["u"],
            "cold_surface_v": cold_surface_obs["v"],
            "cold_surface_w": cold_surface_obs["w"],
            "cold_surface_p": cold_surface_obs["p"],
            "cold_surface_weight": cold_surface_obs["weight"],
            "wall_inner_supervision_xyz": wall_surface_obs["inner_xyz"],
            "wall_inner_supervision_ops": wall_surface_obs["inner_ops"],
            "wall_inner_supervision_T": wall_surface_obs["inner_T"],
            "wall_inner_supervision_qn": wall_surface_obs["inner_qn"],
            "wall_inner_supervision_normal": wall_surface_obs["inner_normal"],
            "wall_inner_supervision_weight": wall_surface_obs["inner_weight"],
            "wall_outer_supervision_xyz": wall_surface_obs["outer_xyz"],
            "wall_outer_supervision_ops": wall_surface_obs["outer_ops"],
            "wall_outer_supervision_T": wall_surface_obs["outer_T"],
            "wall_outer_supervision_qn": wall_surface_obs["outer_qn"],
            "wall_outer_supervision_normal": wall_surface_obs["outer_normal"],
            "wall_outer_supervision_weight": wall_surface_obs["outer_weight"],
            "hot_collocation_xyz": self._tensor(hot_collocation, requires_grad=True),
            "hot_collocation_ops": self._tensor(self._sample_operating_points(len(hot_collocation))),
            "cold_collocation_xyz": self._tensor(cold_collocation, requires_grad=True),
            "cold_collocation_ops": self._tensor(self._sample_operating_points(len(cold_collocation))),
            "wall_collocation_xyz": self._tensor(wall_collocation, requires_grad=True),
            "wall_collocation_ops": self._tensor(self._sample_operating_points(len(wall_collocation))),
            "hot_inlet_xyz": self._tensor(hot_inlet),
            "hot_inlet_ops": self._tensor(self._sample_operating_points(len(hot_inlet))),
            "cold_inlet_xyz": self._tensor(cold_inlet),
            "cold_inlet_ops": self._tensor(self._sample_operating_points(len(cold_inlet))),
            "hot_outlet_xyz": self._tensor(hot_outlet),
            "hot_outlet_ops": self._tensor(self._sample_operating_points(len(hot_outlet))),
            "cold_outlet_xyz": self._tensor(cold_outlet),
            "cold_outlet_ops": self._tensor(self._sample_operating_points(len(cold_outlet))),
            "hot_interface_xyz": self._tensor(hot_interface, requires_grad=True),
            "hot_interface_ops": self._tensor(self._sample_operating_points(len(hot_interface))),
            "wall_inner_interface_xyz": self._tensor(hot_interface.copy(), requires_grad=True),
            "wall_inner_interface_ops": self._tensor(self._sample_operating_points(len(hot_interface))),
            "wall_outer_interface_xyz": self._tensor(cold_interface.copy(), requires_grad=True),
            "wall_outer_interface_ops": self._tensor(self._sample_operating_points(len(cold_interface))),
            "cold_interface_xyz": self._tensor(cold_interface, requires_grad=True),
            "cold_interface_ops": self._tensor(self._sample_operating_points(len(cold_interface))),
        }

    def make_refinement_batch(self, multiplier: float = 2.0) -> dict[str, torch.Tensor]:
        cfg = self.config["training_3d"]
        rng = np.random.default_rng(int(cfg["seed"]) + 10_007)
        hot_surface_obs = self._surface_observation_tensors(self.hot_train)
        cold_surface_obs = self._surface_observation_tensors(self.cold_train)
        wall_surface_obs = self._wall_supervision_tensors(self.hot_train, self.cold_train)

        def scaled_count(key: str) -> int:
            return max(1, int(round(float(cfg[key]) * multiplier)))

        hot_collocation = sample_cylinder_volume(
            scaled_count("collocation_hot"),
            self.geometry.hot_radius_m,
            -self.geometry.hot_half_length_m,
            self.geometry.hot_half_length_m,
            rng,
        )
        cold_collocation = sample_annulus_volume(
            scaled_count("collocation_cold"),
            self.geometry.cold_inner_radius_m,
            self.geometry.cold_outer_radius_m,
            -self.geometry.cold_half_length_m,
            self.geometry.cold_half_length_m,
            rng,
        )
        wall_collocation = sample_annulus_volume(
            scaled_count("collocation_wall"),
            self.geometry.hot_radius_m,
            self.geometry.cold_inner_radius_m,
            -self.geometry.cold_half_length_m,
            self.geometry.cold_half_length_m,
            rng,
        )
        hot_inlet = sample_disk(
            scaled_count("inlet_points_hot"),
            self.geometry.hot_radius_m,
            self.geometry.hot_half_length_m,
            rng,
        )
        cold_inlet = sample_annulus_disk(
            scaled_count("inlet_points_cold"),
            self.geometry.cold_inner_radius_m,
            self.geometry.cold_outer_radius_m,
            -self.geometry.cold_half_length_m,
            rng,
        )
        hot_outlet = sample_disk(
            scaled_count("outlet_points_hot"),
            self.geometry.hot_radius_m,
            -self.geometry.hot_half_length_m,
            rng,
        )
        cold_outlet = sample_annulus_disk(
            scaled_count("outlet_points_cold"),
            self.geometry.cold_inner_radius_m,
            self.geometry.cold_outer_radius_m,
            self.geometry.cold_half_length_m,
            rng,
        )
        hot_interface, cold_interface = sample_interface_pair(
            scaled_count("interface_points"),
            self.geometry.hot_radius_m,
            self.geometry.cold_inner_radius_m,
            -self.geometry.cold_half_length_m,
            self.geometry.cold_half_length_m,
            rng,
        )
        return {
            "hot_surface_xyz": hot_surface_obs["xyz"],
            "hot_surface_ops": hot_surface_obs["ops"],
            "hot_surface_T": hot_surface_obs["T"],
            "hot_surface_u": hot_surface_obs["u"],
            "hot_surface_v": hot_surface_obs["v"],
            "hot_surface_w": hot_surface_obs["w"],
            "hot_surface_p": hot_surface_obs["p"],
            "hot_surface_weight": hot_surface_obs["weight"],
            "cold_surface_xyz": cold_surface_obs["xyz"],
            "cold_surface_ops": cold_surface_obs["ops"],
            "cold_surface_T": cold_surface_obs["T"],
            "cold_surface_u": cold_surface_obs["u"],
            "cold_surface_v": cold_surface_obs["v"],
            "cold_surface_w": cold_surface_obs["w"],
            "cold_surface_p": cold_surface_obs["p"],
            "cold_surface_weight": cold_surface_obs["weight"],
            "wall_inner_supervision_xyz": wall_surface_obs["inner_xyz"],
            "wall_inner_supervision_ops": wall_surface_obs["inner_ops"],
            "wall_inner_supervision_T": wall_surface_obs["inner_T"],
            "wall_inner_supervision_qn": wall_surface_obs["inner_qn"],
            "wall_inner_supervision_normal": wall_surface_obs["inner_normal"],
            "wall_inner_supervision_weight": wall_surface_obs["inner_weight"],
            "wall_outer_supervision_xyz": wall_surface_obs["outer_xyz"],
            "wall_outer_supervision_ops": wall_surface_obs["outer_ops"],
            "wall_outer_supervision_T": wall_surface_obs["outer_T"],
            "wall_outer_supervision_qn": wall_surface_obs["outer_qn"],
            "wall_outer_supervision_normal": wall_surface_obs["outer_normal"],
            "wall_outer_supervision_weight": wall_surface_obs["outer_weight"],
            "hot_collocation_xyz": self._tensor(hot_collocation, requires_grad=True),
            "hot_collocation_ops": self._tensor(self._sample_operating_points(len(hot_collocation))),
            "cold_collocation_xyz": self._tensor(cold_collocation, requires_grad=True),
            "cold_collocation_ops": self._tensor(self._sample_operating_points(len(cold_collocation))),
            "wall_collocation_xyz": self._tensor(wall_collocation, requires_grad=True),
            "wall_collocation_ops": self._tensor(self._sample_operating_points(len(wall_collocation))),
            "hot_inlet_xyz": self._tensor(hot_inlet),
            "hot_inlet_ops": self._tensor(self._sample_operating_points(len(hot_inlet))),
            "cold_inlet_xyz": self._tensor(cold_inlet),
            "cold_inlet_ops": self._tensor(self._sample_operating_points(len(cold_inlet))),
            "hot_outlet_xyz": self._tensor(hot_outlet),
            "hot_outlet_ops": self._tensor(self._sample_operating_points(len(hot_outlet))),
            "cold_outlet_xyz": self._tensor(cold_outlet),
            "cold_outlet_ops": self._tensor(self._sample_operating_points(len(cold_outlet))),
            "hot_interface_xyz": self._tensor(hot_interface, requires_grad=True),
            "hot_interface_ops": self._tensor(self._sample_operating_points(len(hot_interface))),
            "wall_inner_interface_xyz": self._tensor(hot_interface.copy(), requires_grad=True),
            "wall_inner_interface_ops": self._tensor(self._sample_operating_points(len(hot_interface))),
            "wall_outer_interface_xyz": self._tensor(cold_interface.copy(), requires_grad=True),
            "wall_outer_interface_ops": self._tensor(self._sample_operating_points(len(cold_interface))),
            "cold_interface_xyz": self._tensor(cold_interface, requires_grad=True),
            "cold_interface_ops": self._tensor(self._sample_operating_points(len(cold_interface))),
        }

    def _compute_losses(self, batch: dict[str, torch.Tensor], weights: dict[str, float], priors: dict[str, float]) -> dict[str, torch.Tensor]:
        hot_surface_state = self.model.hot(batch["hot_surface_xyz"], batch["hot_surface_ops"])
        cold_surface_state = self.model.cold(batch["cold_surface_xyz"], batch["cold_surface_ops"])
        temp_scale = float(self.config["model_3d"]["temperature_scale_K"])
        velocity_scale = float(
            self.config["training_3d"].get(
                "surface_velocity_scale_m_s",
                self.config["model_3d"].get(
                    "velocity_condition_scale_m_s",
                    max(float(self.operating.inlet_velocity_hot_m_per_s), float(self.operating.inlet_velocity_cold_m_per_s), 1.0),
                ),
            )
        )
        heat_flux_scale = float(self.config["training_3d"].get("surface_heat_flux_scale_w_m2", 5.0e3))
        data_loss = self._weighted_mse(
            (hot_surface_state.T - batch["hot_surface_T"]) / temp_scale,
            torch.zeros_like(batch["hot_surface_T"]),
            batch["hot_surface_weight"],
        )
        data_loss = data_loss + self._weighted_mse(
            (cold_surface_state.T - batch["cold_surface_T"]) / temp_scale,
            torch.zeros_like(batch["cold_surface_T"]),
            batch["cold_surface_weight"],
        )
        surface_velocity_loss = self._weighted_mse(
            (hot_surface_state.u - batch["hot_surface_u"]) / velocity_scale,
            torch.zeros_like(batch["hot_surface_u"]),
            batch["hot_surface_weight"],
        )
        surface_velocity_loss = surface_velocity_loss + self._weighted_mse(
            (hot_surface_state.v - batch["hot_surface_v"]) / velocity_scale,
            torch.zeros_like(batch["hot_surface_v"]),
            batch["hot_surface_weight"],
        )
        surface_velocity_loss = surface_velocity_loss + self._weighted_mse(
            (hot_surface_state.w - batch["hot_surface_w"]) / velocity_scale,
            torch.zeros_like(batch["hot_surface_w"]),
            batch["hot_surface_weight"],
        )
        surface_velocity_loss = surface_velocity_loss + self._weighted_mse(
            (cold_surface_state.u - batch["cold_surface_u"]) / velocity_scale,
            torch.zeros_like(batch["cold_surface_u"]),
            batch["cold_surface_weight"],
        )
        surface_velocity_loss = surface_velocity_loss + self._weighted_mse(
            (cold_surface_state.v - batch["cold_surface_v"]) / velocity_scale,
            torch.zeros_like(batch["cold_surface_v"]),
            batch["cold_surface_weight"],
        )
        surface_velocity_loss = surface_velocity_loss + self._weighted_mse(
            (cold_surface_state.w - batch["cold_surface_w"]) / velocity_scale,
            torch.zeros_like(batch["cold_surface_w"]),
            batch["cold_surface_weight"],
        )
        surface_pressure_loss = self._weighted_mse(
            (hot_surface_state.p - batch["hot_surface_p"]) / self.model.pressure_scale_pa,
            torch.zeros_like(batch["hot_surface_p"]),
            batch["hot_surface_weight"],
        )
        surface_pressure_loss = surface_pressure_loss + self._weighted_mse(
            (cold_surface_state.p - batch["cold_surface_p"]) / self.model.pressure_scale_pa,
            torch.zeros_like(batch["cold_surface_p"]),
            batch["cold_surface_weight"],
        )

        wall_temperature_loss = self._zero_loss()
        wall_heat_flux_loss = self._zero_loss()
        wall_interface_temperature_loss = self._zero_loss()
        if batch["wall_inner_supervision_xyz"].shape[0] > 0:
            inner_wall_state = self.model.wall(
                batch["wall_inner_supervision_xyz"],
                batch["wall_inner_supervision_ops"],
            )
            inner_fluid_state = self.model.hot(
                batch["wall_inner_supervision_xyz"],
                batch["wall_inner_supervision_ops"],
            )
            wall_temperature_loss = wall_temperature_loss + self._weighted_mse(
                (inner_wall_state.T - batch["wall_inner_supervision_T"]) / temp_scale,
                torch.zeros_like(batch["wall_inner_supervision_T"]),
                batch["wall_inner_supervision_weight"],
            )
            wall_interface_temperature_loss = wall_interface_temperature_loss + self._weighted_mse(
                (inner_wall_state.T - inner_fluid_state.T) / temp_scale,
                torch.zeros_like(batch["wall_inner_supervision_T"]),
                batch["wall_inner_supervision_weight"],
            )
            inner_grad = torch.autograd.grad(
                inner_wall_state.T,
                batch["wall_inner_supervision_xyz"],
                grad_outputs=torch.ones_like(inner_wall_state.T),
                create_graph=True,
            )[0]
            inner_flux = -self.model.k_wall * torch.sum(
                inner_grad * batch["wall_inner_supervision_normal"],
                dim=1,
                keepdim=True,
            )
            wall_heat_flux_loss = wall_heat_flux_loss + self._weighted_mse(
                (inner_flux - batch["wall_inner_supervision_qn"]) / heat_flux_scale,
                torch.zeros_like(batch["wall_inner_supervision_qn"]),
                batch["wall_inner_supervision_weight"],
            )
        if batch["wall_outer_supervision_xyz"].shape[0] > 0:
            outer_wall_state = self.model.wall(
                batch["wall_outer_supervision_xyz"],
                batch["wall_outer_supervision_ops"],
            )
            outer_fluid_state = self.model.cold(
                batch["wall_outer_supervision_xyz"],
                batch["wall_outer_supervision_ops"],
            )
            wall_temperature_loss = wall_temperature_loss + self._weighted_mse(
                (outer_wall_state.T - batch["wall_outer_supervision_T"]) / temp_scale,
                torch.zeros_like(batch["wall_outer_supervision_T"]),
                batch["wall_outer_supervision_weight"],
            )
            wall_interface_temperature_loss = wall_interface_temperature_loss + self._weighted_mse(
                (outer_wall_state.T - outer_fluid_state.T) / temp_scale,
                torch.zeros_like(batch["wall_outer_supervision_T"]),
                batch["wall_outer_supervision_weight"],
            )
            outer_grad = torch.autograd.grad(
                outer_wall_state.T,
                batch["wall_outer_supervision_xyz"],
                grad_outputs=torch.ones_like(outer_wall_state.T),
                create_graph=True,
            )[0]
            outer_flux = -self.model.k_wall * torch.sum(
                outer_grad * batch["wall_outer_supervision_normal"],
                dim=1,
                keepdim=True,
            )
            wall_heat_flux_loss = wall_heat_flux_loss + self._weighted_mse(
                (outer_flux - batch["wall_outer_supervision_qn"]) / heat_flux_scale,
                torch.zeros_like(batch["wall_outer_supervision_qn"]),
                batch["wall_outer_supervision_weight"],
            )

        case_bulk_temperature_loss = self._zero_loss()
        case_pressure_drop_loss = self._zero_loss()
        case_heat_duty_loss = self._zero_loss()
        case_stream_heat_duty_loss = self._zero_loss()
        case_energy_balance_loss = self._zero_loss()
        heat_duty_scale = float(self.config["training_3d"].get("case_heat_duty_scale_w", 4.0e3))
        if self.case_aggregate_targets:
            for case_target in self.case_aggregate_targets:
                hot_inlet_state_case = self.model.hot(case_target["hot_inlet_xyz"], case_target["hot_inlet_ops"])
                hot_outlet_state_case = self.model.hot(case_target["hot_outlet_xyz"], case_target["hot_outlet_ops"])
                cold_inlet_state_case = self.model.cold(case_target["cold_inlet_xyz"], case_target["cold_inlet_ops"])
                cold_outlet_state_case = self.model.cold(case_target["cold_outlet_xyz"], case_target["cold_outlet_ops"])
                hot_inlet_bulk_pred = self._weighted_mean_tensor(
                    hot_inlet_state_case.T,
                    case_target["hot_inlet_weight"],
                )
                hot_outlet_bulk_pred = self._weighted_mean_tensor(
                    hot_outlet_state_case.T,
                    case_target["hot_outlet_weight"],
                )
                cold_inlet_bulk_pred = self._weighted_mean_tensor(
                    cold_inlet_state_case.T,
                    case_target["cold_inlet_weight"],
                )
                cold_outlet_bulk_pred = self._weighted_mean_tensor(
                    cold_outlet_state_case.T,
                    case_target["cold_outlet_weight"],
                )
                case_bulk_temperature_loss = case_bulk_temperature_loss + (
                    (hot_outlet_bulk_pred - float(case_target["hot_outlet_bulk_T"])) / temp_scale
                ) ** 2
                case_bulk_temperature_loss = case_bulk_temperature_loss + (
                    (cold_outlet_bulk_pred - float(case_target["cold_outlet_bulk_T"])) / temp_scale
                ) ** 2
                hot_dp_pred = torch.mean(hot_inlet_state_case.p) - torch.mean(hot_outlet_state_case.p)
                cold_dp_pred = torch.mean(cold_inlet_state_case.p) - torch.mean(cold_outlet_state_case.p)
                case_pressure_drop_loss = case_pressure_drop_loss + (
                    (hot_dp_pred - float(case_target["hot_dp_Pa"])) / self.model.pressure_scale_pa
                ) ** 2
                case_pressure_drop_loss = case_pressure_drop_loss + (
                    (cold_dp_pred - float(case_target["cold_dp_Pa"])) / self.model.pressure_scale_pa
                ) ** 2
                q_hot_pred = (
                    float(case_target["hot_mass_flow_kg_s"])
                    * float(self.operating.cp_J_per_kgK)
                    * (float(case_target["hot_inlet_bulk_T"]) - hot_outlet_bulk_pred)
                )
                q_cold_pred = (
                    float(case_target["cold_mass_flow_kg_s"])
                    * float(self.operating.cp_J_per_kgK)
                    * (cold_outlet_bulk_pred - float(case_target["cold_inlet_bulk_T"]))
                )
                q_mean_pred = 0.5 * (q_hot_pred + q_cold_pred)
                case_heat_duty_loss = case_heat_duty_loss + (
                    (q_mean_pred - float(case_target["reference_q_total_w"])) / heat_duty_scale
                ) ** 2
                case_stream_heat_duty_loss = case_stream_heat_duty_loss + (
                    (q_hot_pred - float(case_target["reference_q_total_w"])) / heat_duty_scale
                ) ** 2
                case_stream_heat_duty_loss = case_stream_heat_duty_loss + (
                    (q_cold_pred - float(case_target["reference_q_total_w"])) / heat_duty_scale
                ) ** 2
                case_energy_balance_loss = case_energy_balance_loss + (
                    (q_hot_pred - q_cold_pred) / heat_duty_scale
                ) ** 2
            case_count = max(len(self.case_aggregate_targets), 1)
            case_bulk_temperature_loss = case_bulk_temperature_loss / float(case_count)
            case_pressure_drop_loss = case_pressure_drop_loss / float(case_count)
            case_heat_duty_loss = case_heat_duty_loss / float(case_count)
            case_stream_heat_duty_loss = case_stream_heat_duty_loss / float(case_count)
            case_energy_balance_loss = case_energy_balance_loss / float(case_count)

        hot_res = fluid_residuals(
            self.model.hot(batch["hot_collocation_xyz"], batch["hot_collocation_ops"]),
            batch["hot_collocation_xyz"],
            rho_kg_m3=float(self.operating.density_kg_per_m3),
            nu_m2_s=self.model.nu_hot,
            alpha_m2_s=self.model.alpha_hot,
        )
        cold_res = fluid_residuals(
            self.model.cold(batch["cold_collocation_xyz"], batch["cold_collocation_ops"]),
            batch["cold_collocation_xyz"],
            rho_kg_m3=float(self.operating.density_kg_per_m3),
            nu_m2_s=self.model.nu_cold,
            alpha_m2_s=self.model.alpha_cold,
        )
        hot_length = float(self.geometry.hot_half_length_m)
        cold_length = float(self.geometry.cold_half_length_m)
        hot_velocity = float(
            np.max(self.training_case_bank[:, 2]) if len(self.training_case_bank) > 0 else self.operating.inlet_velocity_hot_m_per_s
        )
        cold_velocity = float(
            np.max(self.training_case_bank[:, 3]) if len(self.training_case_bank) > 0 else self.operating.inlet_velocity_cold_m_per_s
        )
        hot_cont_scale = max(hot_velocity / hot_length, 1.0e-6)
        cold_cont_scale = max(cold_velocity / cold_length, 1.0e-6)
        hot_mom_scale = max(hot_velocity**2 / hot_length, 1.0e-6)
        cold_mom_scale = max(cold_velocity**2 / cold_length, 1.0e-6)
        hot_energy_scale = max(hot_velocity * temp_scale / hot_length, 1.0e-6)
        cold_energy_scale = max(cold_velocity * temp_scale / cold_length, 1.0e-6)
        pde_loss = torch.mean((hot_res["continuity"] / hot_cont_scale) ** 2)
        pde_loss = pde_loss + torch.mean((cold_res["continuity"] / cold_cont_scale) ** 2)
        pde_loss = pde_loss + torch.mean((hot_res["momentum_u"] / hot_mom_scale) ** 2)
        pde_loss = pde_loss + torch.mean((hot_res["momentum_v"] / hot_mom_scale) ** 2)
        pde_loss = pde_loss + torch.mean((hot_res["momentum_w"] / hot_mom_scale) ** 2)
        pde_loss = pde_loss + torch.mean((cold_res["momentum_u"] / cold_mom_scale) ** 2)
        pde_loss = pde_loss + torch.mean((cold_res["momentum_v"] / cold_mom_scale) ** 2)
        pde_loss = pde_loss + torch.mean((cold_res["momentum_w"] / cold_mom_scale) ** 2)
        pde_loss = pde_loss + torch.mean((hot_res["energy"] / hot_energy_scale) ** 2)
        pde_loss = pde_loss + torch.mean((cold_res["energy"] / cold_energy_scale) ** 2)
        wall_res = wall_residuals(
            self.model.wall(batch["wall_collocation_xyz"], batch["wall_collocation_ops"]),
            batch["wall_collocation_xyz"],
        )
        wall_thickness = max(float(self.geometry.cold_inner_radius_m - self.geometry.hot_radius_m), 1.0e-6)
        wall_scale = max(temp_scale / (wall_thickness**2), 1.0e-6)
        wall_pde_loss = torch.mean((wall_res["conduction"] / wall_scale) ** 2)

        hot_inlet_state = self.model.hot(batch["hot_inlet_xyz"], batch["hot_inlet_ops"])
        cold_inlet_state = self.model.cold(batch["cold_inlet_xyz"], batch["cold_inlet_ops"])
        hot_inlet_factor = self.model.hot_profile_factor(batch["hot_inlet_xyz"])
        cold_inlet_factor = self.model.cold_profile_factor(batch["cold_inlet_xyz"])
        if self.condition_on_operating_point:
            hot_inlet_velocity = batch["hot_inlet_ops"][:, 2:3]
            cold_inlet_velocity = batch["cold_inlet_ops"][:, 3:4]
            hot_inlet_target = batch["hot_inlet_ops"][:, 0:1] + self.model.hot_inlet_temperature_bias
            cold_inlet_target = batch["cold_inlet_ops"][:, 1:2] + self.model.cold_inlet_temperature_bias
        else:
            hot_inlet_velocity = torch.full_like(hot_inlet_factor, float(self.operating.inlet_velocity_hot_m_per_s))
            cold_inlet_velocity = torch.full_like(cold_inlet_factor, float(self.operating.inlet_velocity_cold_m_per_s))
            hot_inlet_target = torch.full_like(hot_inlet_state.T, float(self.hot_inlet_target_K))
            cold_inlet_target = torch.full_like(cold_inlet_state.T, float(self.cold_inlet_target_K))
        inlet_loss = torch.mean(hot_inlet_state.u**2 + hot_inlet_state.v**2)
        inlet_loss = inlet_loss + torch.mean(cold_inlet_state.u**2 + cold_inlet_state.v**2)
        inlet_loss = inlet_loss + torch.mean(
            (hot_inlet_state.w + hot_inlet_velocity * hot_inlet_factor) ** 2
        )
        inlet_loss = inlet_loss + torch.mean(
            (cold_inlet_state.w - cold_inlet_velocity * cold_inlet_factor) ** 2
        )
        inlet_loss = inlet_loss + torch.mean(((hot_inlet_state.T - hot_inlet_target) / temp_scale) ** 2)
        inlet_loss = inlet_loss + torch.mean(((cold_inlet_state.T - cold_inlet_target) / temp_scale) ** 2)

        hot_outlet_state = self.model.hot(batch["hot_outlet_xyz"], batch["hot_outlet_ops"])
        cold_outlet_state = self.model.cold(batch["cold_outlet_xyz"], batch["cold_outlet_ops"])
        outlet_loss = torch.mean((hot_outlet_state.p / self.model.pressure_scale_pa) ** 2)
        outlet_loss = outlet_loss + torch.mean((cold_outlet_state.p / self.model.pressure_scale_pa) ** 2)

        interface_res = conjugate_interface_residuals(
            self.model,
            batch["hot_interface_xyz"],
            batch["wall_inner_interface_xyz"],
            batch["wall_outer_interface_xyz"],
            batch["cold_interface_xyz"],
            rho_kg_m3=float(self.operating.density_kg_per_m3),
            cp_J_kgK=float(self.operating.cp_J_per_kgK),
            hot_operating_point=batch["hot_interface_ops"],
            wall_inner_operating_point=batch["wall_inner_interface_ops"],
            wall_outer_operating_point=batch["wall_outer_interface_ops"],
            cold_operating_point=batch["cold_interface_ops"],
        )
        interface_temp_loss = torch.mean((interface_res["temp_hot_wall"] / temp_scale) ** 2)
        interface_temp_loss = interface_temp_loss + torch.mean((interface_res["temp_wall_cold"] / temp_scale) ** 2)
        interface_flux_loss = torch.mean((interface_res["flux_hot_wall"] / 1.0e4) ** 2)
        interface_flux_loss = interface_flux_loss + torch.mean((interface_res["flux_wall_cold"] / 1.0e4) ** 2)
        ordering_loss = torch.mean((interface_res["ordering"] / temp_scale) ** 2)

        transport_reg_loss = ((self.model.nu_hot - priors["nu_hot"]) / priors["nu_hot"]) ** 2
        transport_reg_loss = transport_reg_loss + ((self.model.nu_cold - priors["nu_cold"]) / priors["nu_cold"]) ** 2
        transport_reg_loss = transport_reg_loss + ((self.model.alpha_hot - priors["alpha_hot"]) / priors["alpha_hot"]) ** 2
        transport_reg_loss = transport_reg_loss + ((self.model.alpha_cold - priors["alpha_cold"]) / priors["alpha_cold"]) ** 2
        wall_reg_loss = ((self.model.k_wall - priors["k_wall"]) / priors["k_wall"]) ** 2

        total = (
            weights["data_weight"] * data_loss
            + weights["surface_velocity_weight"] * surface_velocity_loss
            + weights["surface_pressure_weight"] * surface_pressure_loss
            + weights["wall_temperature_weight"] * wall_temperature_loss
            + weights["wall_heat_flux_weight"] * wall_heat_flux_loss
            + weights["wall_interface_temperature_weight"] * wall_interface_temperature_loss
            + weights["case_bulk_temperature_weight"] * case_bulk_temperature_loss
            + weights["case_pressure_drop_weight"] * case_pressure_drop_loss
            + weights["case_heat_duty_weight"] * case_heat_duty_loss
            + weights["case_stream_heat_duty_weight"] * case_stream_heat_duty_loss
            + weights["case_energy_balance_weight"] * case_energy_balance_loss
            + weights["pde_weight"] * pde_loss
            + weights["wall_pde_weight"] * wall_pde_loss
            + weights["inlet_weight"] * inlet_loss
            + weights["outlet_weight"] * outlet_loss
            + weights["interface_temp_weight"] * interface_temp_loss
            + weights["interface_flux_weight"] * interface_flux_loss
            + weights["ordering_weight"] * ordering_loss
            + weights["reg_weight"] * transport_reg_loss
            + weights["wall_reg_weight"] * wall_reg_loss
        )
        return {
            "total": total,
            "data": data_loss,
            "surface_velocity": surface_velocity_loss,
            "surface_pressure": surface_pressure_loss,
            "wall_temperature": wall_temperature_loss,
            "wall_heat_flux": wall_heat_flux_loss,
            "wall_interface_temperature": wall_interface_temperature_loss,
            "case_bulk_temperature": case_bulk_temperature_loss,
            "case_pressure_drop": case_pressure_drop_loss,
            "case_heat_duty": case_heat_duty_loss,
            "case_stream_heat_duty": case_stream_heat_duty_loss,
            "case_energy_balance": case_energy_balance_loss,
            "pde": pde_loss,
            "wall_pde": wall_pde_loss,
            "inlet": inlet_loss,
            "outlet": outlet_loss,
            "interface_temp": interface_temp_loss,
            "interface_flux": interface_flux_loss,
            "ordering": ordering_loss,
            "reg": transport_reg_loss,
            "wall_reg": wall_reg_loss,
        }

    def fit(self, output_dir: str | Path, adam_epochs_override: int | None = None) -> Trainer3DArtifacts:
        output_path = Path(output_dir)
        (output_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_path / "figures").mkdir(parents=True, exist_ok=True)
        training_cfg = self.config["training_3d"]
        model_cfg = self.config["model_3d"]
        weights = {
            "data_weight": float(training_cfg["data_weight"]),
            "surface_velocity_weight": float(training_cfg.get("surface_velocity_weight", 0.0)),
            "surface_pressure_weight": float(training_cfg.get("surface_pressure_weight", 0.0)),
            "wall_temperature_weight": float(training_cfg.get("wall_temperature_weight", 0.0)),
            "wall_heat_flux_weight": float(training_cfg.get("wall_heat_flux_weight", 0.0)),
            "wall_interface_temperature_weight": float(training_cfg.get("wall_interface_temperature_weight", 0.0)),
            "case_bulk_temperature_weight": float(training_cfg.get("case_bulk_temperature_weight", 0.0)),
            "case_pressure_drop_weight": float(training_cfg.get("case_pressure_drop_weight", 0.0)),
            "case_heat_duty_weight": float(training_cfg.get("case_heat_duty_weight", 0.0)),
            "case_stream_heat_duty_weight": float(training_cfg.get("case_stream_heat_duty_weight", 0.0)),
            "case_energy_balance_weight": float(training_cfg.get("case_energy_balance_weight", 0.0)),
            "pde_weight": float(training_cfg["pde_weight"]),
            "wall_pde_weight": float(training_cfg["wall_pde_weight"]),
            "inlet_weight": float(training_cfg["inlet_weight"]),
            "outlet_weight": float(training_cfg["outlet_weight"]),
            "interface_temp_weight": float(training_cfg["interface_temp_weight"]),
            "interface_flux_weight": float(training_cfg["interface_flux_weight"]),
            "ordering_weight": float(training_cfg["ordering_weight"]),
            "reg_weight": float(training_cfg["reg_weight"]),
            "wall_reg_weight": float(training_cfg.get("wall_reg_weight", training_cfg["reg_weight"])),
        }
        priors = {
            "nu_hot": float(model_cfg["initial_nu_hot_m2_s"]),
            "nu_cold": float(model_cfg["initial_nu_cold_m2_s"]),
            "alpha_hot": float(model_cfg["initial_alpha_hot_m2_s"]),
            "alpha_cold": float(model_cfg["initial_alpha_cold_m2_s"]),
            "k_wall": float(model_cfg["initial_k_wall_w_mk"]),
        }
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(training_cfg["adam_lr"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            min_lr=1.0e-5,
        )

        history: list[dict[str, float]] = []
        best_score = float("inf")
        checkpoint_path = output_path / "checkpoints" / "best_model_3d.pt"
        epochs = int(adam_epochs_override if adam_epochs_override is not None else training_cfg["adam_epochs"])
        for epoch in range(1, epochs + 1):
            batch = self.make_batch()
            optimizer.zero_grad()
            losses = self._compute_losses(batch, self._phase_weights(epoch, epochs, weights), priors)
            losses["total"].backward()
            clip_grad_norm_(self.model.parameters(), max_norm=float(training_cfg["grad_clip"]))
            optimizer.step()
            validation = self._compute_validation_objectives()
            validation_score = validation["score"]
            scheduler.step(validation_score)
            record = {name: float(value.detach().cpu()) for name, value in losses.items()}
            record["epoch"] = epoch
            record["validation_score"] = validation_score
            record["validation_surface_rmse"] = validation["surface_rmse_K"]
            record["validation_q_rel_pct"] = validation["mean_q_rel_error_pct"]
            record["validation_energy_gap_pct"] = validation["mean_energy_gap_pct"]
            record["validation_dp_rel_pct"] = validation["mean_dp_rel_error_pct"]
            record["validation_wall_target_rmse"] = validation["mean_wall_target_rmse_K"]
            record["validation_wall_stream_gap_rmse"] = validation["mean_wall_stream_gap_rmse_K"]
            record["nu_hot_m2_s"] = float(self.model.nu_hot.detach().cpu())
            record["nu_cold_m2_s"] = float(self.model.nu_cold.detach().cpu())
            record["alpha_hot_m2_s"] = float(self.model.alpha_hot.detach().cpu())
            record["alpha_cold_m2_s"] = float(self.model.alpha_cold.detach().cpu())
            record["k_wall_w_mk"] = float(self.model.k_wall.detach().cpu())
            record.update(self._resolved_inlet_temperature_summary())
            history.append(record)
            if validation_score < best_score:
                best_score = validation_score
                torch.save(self.model.state_dict(), checkpoint_path)
            if epoch == 1 or epoch % int(training_cfg["print_every"]) == 0 or epoch == epochs:
                print(
                    f"epoch={epoch:05d} total={record['total']:.6f} data={record['data']:.6f} "
                    f"vel={record['surface_velocity']:.6f} p={record['surface_pressure']:.6f} "
                    f"wallT={record['wall_temperature']:.6f} wallI={record['wall_interface_temperature']:.6f} "
                    f"flux={record['wall_heat_flux']:.6f} "
                    f"bulk={record['case_bulk_temperature']:.6f} dp={record['case_pressure_drop']:.6f} "
                    f"Q={record['case_heat_duty']:.6f} Qs={record['case_stream_heat_duty']:.6f} "
                    f"Eb={record['case_energy_balance']:.6f} pde={record['pde']:.6f} "
                    f"val={validation_score:.4f} surf={record['validation_surface_rmse']:.4f} "
                    f"Qv={record['validation_q_rel_pct']:.2f}% Ebv={record['validation_energy_gap_pct']:.2f}% "
                    f"DPv={record['validation_dp_rel_pct']:.2f}% Wv={record['validation_wall_target_rmse']:.2f} "
                    f"kwall={record['k_wall_w_mk']:.3f} "
                    f"Th_in={record['hot_inlet_temperature_K']:.2f} "
                    f"Tc_in={record['cold_inlet_temperature_K']:.2f} "
                    f"dTh={record['hot_inlet_temperature_bias_K']:.2f} "
                    f"dTc={record['cold_inlet_temperature_bias_K']:.2f}"
                )

        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        lbfgs_steps = int(training_cfg.get("lbfgs_steps", 0))
        if lbfgs_steps > 0:
            refinement_batch = self.make_refinement_batch(
                multiplier=float(training_cfg.get("lbfgs_collocation_multiplier", 2.0))
            )
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=float(training_cfg.get("lbfgs_lr", 0.5)),
                max_iter=int(training_cfg.get("lbfgs_max_iter", 20)),
                history_size=int(training_cfg.get("lbfgs_history_size", 50)),
                line_search_fn="strong_wolfe",
            )
            for step in range(1, lbfgs_steps + 1):
                def closure() -> torch.Tensor:
                    lbfgs.zero_grad()
                    lbfgs_losses = self._compute_losses(refinement_batch, weights, priors)
                    lbfgs_losses["total"].backward()
                    return lbfgs_losses["total"]

                lbfgs.step(closure)
                losses = self._compute_losses(refinement_batch, weights, priors)
                validation = self._compute_validation_objectives()
                validation_score = validation["score"]
                record = {name: float(value.detach().cpu()) for name, value in losses.items()}
                record["epoch"] = epochs + step
                record["validation_score"] = validation_score
                record["validation_surface_rmse"] = validation["surface_rmse_K"]
                record["validation_q_rel_pct"] = validation["mean_q_rel_error_pct"]
                record["validation_energy_gap_pct"] = validation["mean_energy_gap_pct"]
                record["validation_dp_rel_pct"] = validation["mean_dp_rel_error_pct"]
                record["validation_wall_target_rmse"] = validation["mean_wall_target_rmse_K"]
                record["validation_wall_stream_gap_rmse"] = validation["mean_wall_stream_gap_rmse_K"]
                record["nu_hot_m2_s"] = float(self.model.nu_hot.detach().cpu())
                record["nu_cold_m2_s"] = float(self.model.nu_cold.detach().cpu())
                record["alpha_hot_m2_s"] = float(self.model.alpha_hot.detach().cpu())
                record["alpha_cold_m2_s"] = float(self.model.alpha_cold.detach().cpu())
                record["k_wall_w_mk"] = float(self.model.k_wall.detach().cpu())
                record.update(self._resolved_inlet_temperature_summary())
                history.append(record)
                if validation_score < best_score:
                    best_score = validation_score
                    torch.save(self.model.state_dict(), checkpoint_path)
                print(
                    f"lbfgs={step:03d} total={record['total']:.6f} data={record['data']:.6f} "
                    f"vel={record['surface_velocity']:.6f} p={record['surface_pressure']:.6f} "
                    f"wallT={record['wall_temperature']:.6f} wallI={record['wall_interface_temperature']:.6f} "
                    f"flux={record['wall_heat_flux']:.6f} "
                    f"bulk={record['case_bulk_temperature']:.6f} dp={record['case_pressure_drop']:.6f} "
                    f"Q={record['case_heat_duty']:.6f} Qs={record['case_stream_heat_duty']:.6f} "
                    f"Eb={record['case_energy_balance']:.6f} pde={record['pde']:.6f} "
                    f"val={validation_score:.4f} surf={record['validation_surface_rmse']:.4f} "
                    f"Qv={record['validation_q_rel_pct']:.2f}% Ebv={record['validation_energy_gap_pct']:.2f}% "
                    f"DPv={record['validation_dp_rel_pct']:.2f}% Wv={record['validation_wall_target_rmse']:.2f} "
                    f"kwall={record['k_wall_w_mk']:.3f} "
                    f"Th_in={record['hot_inlet_temperature_K']:.2f} "
                    f"Tc_in={record['cold_inlet_temperature_K']:.2f} "
                    f"dTh={record['hot_inlet_temperature_bias_K']:.2f} "
                    f"dTc={record['cold_inlet_temperature_bias_K']:.2f}"
                )

            if checkpoint_path.exists():
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        predictions = self.predict_surface_temperatures()
        predictions_path = output_path / "surface_predictions_3d.csv"
        predictions.to_csv(predictions_path, index=False)
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(output_path / "training_history_3d.csv", index=False)
        inlet_summary = self._resolved_inlet_temperature_summary()
        metrics = {
            "best_validation_score": best_score,
            "conditioning_enabled": self.condition_on_operating_point,
            "training_case_count": int(len(self.training_case_bank)),
            "train_case_ids": self._case_ids(self.hot_train),
            "validation_case_ids": self._case_ids(self.hot_validation),
            "hot_inlet_target_K": float(self.hot_inlet_target_K),
            "cold_inlet_target_K": float(self.cold_inlet_target_K),
            "hot_inlet_temperature_source": str(training_cfg.get("hot_inlet_temperature_source", "reference")),
            "cold_inlet_temperature_source": str(training_cfg.get("cold_inlet_temperature_source", "train_boundary_mean")),
            "final_nu_hot_m2_s": float(self.model.nu_hot.detach().cpu()),
            "final_nu_cold_m2_s": float(self.model.nu_cold.detach().cpu()),
            "final_alpha_hot_m2_s": float(self.model.alpha_hot.detach().cpu()),
            "final_alpha_cold_m2_s": float(self.model.alpha_cold.detach().cpu()),
            "final_k_wall_w_mk": float(self.model.k_wall.detach().cpu()),
            "final_hot_inlet_temperature_K": inlet_summary["hot_inlet_temperature_K"],
            "final_cold_inlet_temperature_K": inlet_summary["cold_inlet_temperature_K"],
            "final_hot_inlet_temperature_bias_K": inlet_summary["hot_inlet_temperature_bias_K"],
            "final_cold_inlet_temperature_bias_K": inlet_summary["cold_inlet_temperature_bias_K"],
            "epochs": len(history),
            "train_metrics": self._compute_metrics(self.hot_train, self.cold_train),
            "validation_metrics": self._compute_metrics(self.hot_validation, self.cold_validation),
        }
        metrics_path = output_path / "training_metrics_3d.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        self._save_plots(predictions, history_frame, output_path / "figures")
        return Trainer3DArtifacts(
            history=history,
            best_score=best_score,
            checkpoint_path=checkpoint_path,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
        )

    def _predict_temperatures(self, frame: pd.DataFrame, stream: str) -> np.ndarray:
        xyz = self._tensor(self._xyz_from_frame(frame))
        operating_point = self._tensor(self._ops_from_frame(frame))
        with torch.no_grad():
            state = self.model.hot(xyz, operating_point) if stream == "hot" else self.model.cold(xyz, operating_point)
        return state.T.detach().cpu().numpy().reshape(-1)

    @staticmethod
    def _compute_stream_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | dict[str, dict[str, float]]]:
        target = frame["T"].to_numpy(dtype=np.float64)
        result: dict[str, float | dict[str, dict[str, float]]] = {
            "rmse_K": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "mae_K": float(np.mean(np.abs(prediction - target))),
            "by_boundary": {},
        }
        boundary_metrics: dict[str, dict[str, float]] = {}
        for boundary, group in frame.groupby("boundary", sort=False):
            idx = group.index.to_numpy(dtype=np.int64)
            group_target = target[idx]
            group_pred = prediction[idx]
            boundary_metrics[str(boundary)] = {
                "rmse_K": float(np.sqrt(np.mean((group_pred - group_target) ** 2))),
                "mae_K": float(np.mean(np.abs(group_pred - group_target))),
                "mean_target_K": float(np.mean(group_target)),
                "mean_pred_K": float(np.mean(group_pred)),
            }
        result["by_boundary"] = boundary_metrics
        if "case_id" in frame.columns:
            case_metrics: dict[str, dict[str, float]] = {}
            for case_id, group in frame.groupby("case_id", sort=False):
                idx = group.index.to_numpy(dtype=np.int64)
                group_target = target[idx]
                group_pred = prediction[idx]
                case_metrics[str(case_id)] = {
                    "rmse_K": float(np.sqrt(np.mean((group_pred - group_target) ** 2))),
                    "mae_K": float(np.mean(np.abs(group_pred - group_target))),
                    "mean_target_K": float(np.mean(group_target)),
                    "mean_pred_K": float(np.mean(group_pred)),
                }
            result["by_case"] = case_metrics
        return result

    def _compute_metrics(self, hot_frame: pd.DataFrame, cold_frame: pd.DataFrame) -> dict[str, dict[str, float | dict[str, dict[str, float]]]]:
        hot_pred = self._predict_temperatures(hot_frame, "hot")
        cold_pred = self._predict_temperatures(cold_frame, "cold")
        return {
            "hot": self._compute_stream_metrics(hot_frame, hot_pred),
            "cold": self._compute_stream_metrics(cold_frame, cold_pred),
        }

    def _compute_validation_metrics(self) -> dict[str, dict[str, float]]:
        return self._compute_metrics(self.hot_validation, self.cold_validation)

    def predict_surface_temperatures(self) -> pd.DataFrame:
        hot_all = self.hot_train.assign(dataset="train")
        hot_all = pd.concat([hot_all, self.hot_validation.assign(dataset="validation")], ignore_index=True)
        cold_all = self.cold_train.assign(dataset="train")
        cold_all = pd.concat([cold_all, self.cold_validation.assign(dataset="validation")], ignore_index=True)
        hot_pred = self._predict_temperatures(hot_all, "hot")
        cold_pred = self._predict_temperatures(cold_all, "cold")
        hot_frame = hot_all.copy()
        hot_frame["stream"] = "hot"
        hot_frame["T_pred"] = hot_pred
        cold_frame = cold_all.copy()
        cold_frame["stream"] = "cold"
        cold_frame["T_pred"] = cold_pred
        return pd.concat([hot_frame, cold_frame], ignore_index=True)

    def _save_plots(self, predictions: pd.DataFrame, history: pd.DataFrame, figures_dir: Path) -> None:
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.scatter(predictions["T"], predictions["T_pred"], s=10, alpha=0.4)
        lower = min(predictions["T"].min(), predictions["T_pred"].min())
        upper = max(predictions["T"].max(), predictions["T_pred"].max())
        plt.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
        plt.xlabel("Measured surface temperature [K]")
        plt.ylabel("Predicted surface temperature [K]")
        plt.title("3D PINN surface-temperature parity")
        plt.tight_layout()
        plt.savefig(figures_dir / "surface_parity_3d.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        plotted = False
        for column in [
            "total",
            "data",
            "surface_velocity",
            "surface_pressure",
            "wall_temperature",
            "wall_heat_flux",
            "wall_interface_temperature",
            "case_bulk_temperature",
            "case_pressure_drop",
            "case_heat_duty",
            "case_stream_heat_duty",
            "case_energy_balance",
            "pde",
            "wall_pde",
            "inlet",
            "outlet",
            "interface_temp",
            "interface_flux",
            "ordering",
            "reg",
            "wall_reg",
        ]:
            if column in history.columns:
                plt.plot(history["epoch"], history[column], label=column)
                plotted = True
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("3D multiphysics PINN training losses")
        if plotted:
            plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "training_loss_3d.png", dpi=200)
        plt.close()

    def _phase_weights(self, epoch: int, total_epochs: int, base: dict[str, float]) -> dict[str, float]:
        weights = dict(base)
        warmup_epochs = int(self.config["training_3d"].get("warmup_epochs", 0))
        if warmup_epochs > 0 and epoch <= min(warmup_epochs, total_epochs):
            weights["pde_weight"] *= float(self.config["training_3d"].get("warmup_pde_scale", 1.0))
            weights["wall_pde_weight"] *= float(self.config["training_3d"].get("warmup_wall_pde_scale", 1.0))
            weights["interface_flux_weight"] *= float(
                self.config["training_3d"].get("warmup_interface_flux_scale", 1.0)
            )
        return weights
