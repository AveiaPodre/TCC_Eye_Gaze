"""
Procedimentos com alvos em tela cheia: calibração, validação e recentralização.

Cada procedimento é uma máquina de estados atualizada a cada frame pelo
loop principal (``update``) e desenhada numa imagem do tamanho da tela
(``render``). Assim a câmera e o rastreamento continuam rodando
normalmente durante o procedimento, sem loops aninhados.

Para cada alvo:
    1. "settle"  - o alvo aparece e encolhe (atrai o olhar); nada é coletado.
    2. "collect" - amostras válidas (rosto visível, sem piscada) são guardadas.
       Se o rosto sumir, o alvo espera até juntar o mínimo de amostras.

Cada amostra guarda duas coisas: os ângulos (yaw, pitch), usados para
rejeitar outliers e no CSV, e o vetor bruto ``raw``, entrada do modelo de
calibração (ver calibration.py).

A calibração tem uma etapa extra no fim: o alvo fica no centro e setas
pedem que a pessoa gire a cabeça devagar (esquerda, direita, cima, baixo)
sem tirar o olho do alvo.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import (CalibrationData, CalibrationModel, PointResult, build_grid,
                          fit_from_samples, outlier_mask)
from .config import CalibrationConfig, ValidationConfig
from .gaze import GazeSample

BG = (40, 40, 40)
WHITE = (240, 240, 240)
TARGET = (0, 170, 255)
COLLECT = (80, 220, 80)
WARN = (60, 60, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.8, color=WHITE, thick=2):
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 5), (x + tw + 4, y + base + 1), (15, 15, 15), -1)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class TargetProcedure:
    title = "Procedimento"
    result_timeout_s = 8.0

    def __init__(self, targets, screen_size, settle_s, collect_s, min_samples):
        self.targets = [tuple(map(float, t)) for t in targets]
        self.screen_w, self.screen_h = screen_size
        self.settle_s = settle_s
        self.collect_s = collect_s
        self.min_samples = min_samples
        # Por alvo: lista de (features (2,), raw (14,)).
        self.samples: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in self.targets]
        # Landmarks de cada amostra (para recalcular o olhar offline).
        self.observations: list[list] = [[] for _ in self.targets]
        self.index = 0
        self._phase_start: float | None = None
        self._last_valid = 0.0
        self.collecting_done = False   # todos os alvos coletados
        self.finished = False          # usuário já viu o resultado
        self.succeeded = False
        self.message_lines: list[str] = []
        self._result_since: float | None = None
        # Coleta terminou; o ajuste roda no próximo frame (depois de a tela
        # "Calculando..." ter sido exibida).
        self.computing = False

    # ------------------------------------------------------------ estado
    def _run_pending(self, now: float) -> bool:
        if not self.computing:
            return False
        self.on_complete()
        self.computing = False
        self.collecting_done = True
        self._result_since = now
        return True

    def update(self, sample: GazeSample | None, now: float):
        if self._run_pending(now):
            return
        if self.collecting_done:
            if self._result_since is None:
                self._result_since = now
            elif now - self._result_since > self.result_timeout_s:
                self.finished = True
            return

        if self._phase_start is None:
            self._phase_start = now
            self._last_valid = now
        elapsed = now - self._phase_start

        if sample is not None and sample.valid:
            self._last_valid = now
            if elapsed >= self.settle_s:
                self.samples[self.index].append((sample.features(), sample.raw))
                self.observations[self.index].append(sample.obs)

        enough = len(self.samples[self.index]) >= self.min_samples
        if elapsed >= self.settle_s + self.collect_s and enough:
            self.index += 1
            self._phase_start = now
            if self.index >= len(self.targets):
                self.targets_done(now)

    def targets_done(self, now: float):
        """Todos os alvos coletados (subclasses podem acrescentar etapas)."""
        self.computing = True

    def arrays(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(features (n,2), raw (n,14)) do alvo i."""
        feats = np.array([f for f, _ in self.samples[i]])
        raws = np.array([r for _, r in self.samples[i]])
        return feats, raws

    def current_target(self) -> tuple[float, float]:
        return self.targets[min(self.index, len(self.targets) - 1)]

    def acknowledge(self):
        """Chamado quando o usuário aperta uma tecla na tela de resultado."""
        if self.collecting_done:
            self.finished = True

    def on_complete(self):  # sobrescrito nas subclasses
        self.succeeded = True

    # ----------------------------------------------------------- desenho
    def render(self, now: float) -> np.ndarray:
        canvas = np.full((self.screen_h, self.screen_w, 3), BG, dtype=np.uint8)
        if self.computing:
            self._centered(canvas, f"{self.title}: calculando...", self.screen_h // 2)
            return canvas
        if self.collecting_done:
            self.render_result(canvas)
            _text(canvas, "Pressione qualquer tecla para continuar",
                  (30, self.screen_h - 30), 0.7)
            return canvas

        tx, ty = (int(round(v)) for v in self.targets[self.index])
        elapsed = 0.0 if self._phase_start is None else now - self._phase_start

        if elapsed < self.settle_s:
            k = elapsed / self.settle_s
            r = int(40 - 28 * k)
            cv2.circle(canvas, (tx, ty), r, TARGET, 3, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 4, WHITE, -1, cv2.LINE_AA)
        else:
            n = len(self.samples[self.index])
            k = min(1.0, (elapsed - self.settle_s) / self.collect_s)
            k = min(k, n / self.min_samples) if n < self.min_samples else k
            cv2.circle(canvas, (tx, ty), 12, COLLECT, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, 360 * k, COLLECT, 3, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 3, (0, 0, 0), -1, cv2.LINE_AA)

        # Texto no lado oposto ao alvo, para não competir com ele pelo olhar.
        top = ty >= self.screen_h / 2
        y0 = 45 if top else self.screen_h - 75
        self._centered(canvas, f"{self.title}: olhe para o ponto "
                               f"({self.index + 1}/{len(self.targets)})  -  ESC cancela", y0)
        if now - self._last_valid > 1.0:
            self._centered(canvas, "Rosto/olhos nao detectados - posicione-se em frente a camera",
                           y0 + 40, WARN)
        return canvas

    def _centered(self, canvas, s, y, color=WHITE):
        (tw, _), _ = cv2.getTextSize(s, FONT, 0.8, 2)
        _text(canvas, s, ((self.screen_w - tw) // 2, y), 0.8, color)

    def render_result(self, canvas: np.ndarray):
        """Escreve as mensagens num bloco centralizado que não cubra nenhum alvo."""
        if not self.message_lines:
            return
        scale, step = 0.8, 40
        widths = [cv2.getTextSize(l, FONT, scale, 2)[0][0] for l in self.message_lines]
        bw, bh = max(widths) + 20, step * len(self.message_lines) + 10
        x0 = (self.screen_w - bw) // 2
        margin = 35
        candidates = [int(f * self.screen_h)
                      for f in (0.18, 0.62, 0.25, 0.70, 0.12, 0.30, 0.01, 0.80)]
        y0 = candidates[0]
        for cy in candidates:
            if cy + bh > self.screen_h - 60:
                continue
            if all(not (x0 - margin < tx < x0 + bw + margin and cy - margin < ty < cy + bh + margin)
                   for tx, ty in self.targets):
                y0 = cy
                break
        for i, line in enumerate(self.message_lines):
            _text(canvas, line, (x0 + 10, y0 + 30 + step * i), scale)


def _draw_point_results(canvas, results: list[PointResult], tol_px: float):
    for pr in results:
        t = tuple(int(round(v)) for v in pr.target)
        p = tuple(int(round(v)) for v in pr.prediction)
        color = COLLECT if pr.error_px <= tol_px else WARN
        cv2.circle(canvas, t, 14, WHITE, 2, cv2.LINE_AA)
        cv2.line(canvas, t, p, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p, 7, color, -1, cv2.LINE_AA)
        _text(canvas, f"{pr.error_px:.0f}px", (t[0] + 18, t[1] - 18), 0.6)


HEAD_PROMPTS = (
    ("ESQUERDA", (-1, 0)),
    ("DIREITA", (1, 0)),
    ("CIMA", (0, -1)),
    ("BAIXO", (0, 1)),
    ("CENTRO", (0, 0)),
)


class CalibrationProcedure(TargetProcedure):
    title = "Calibracao"

    def __init__(self, screen_size, cfg: CalibrationConfig):
        targets = build_grid(*screen_size, cfg.grid_cols, cfg.grid_rows, cfg.margin_ratio)
        super().__init__(targets, screen_size, cfg.settle_s, cfg.collect_s,
                         cfg.min_samples_per_point)
        self.cfg = cfg
        self.model: CalibrationModel | None = None
        self.raw_path: Path | None = None
        self.results: list[PointResult] = []
        self.stats: dict = {}
        self.head_start: float | None = None
        self.head_samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.head_observations: list = []
        self._now = 0.0

    # ----------------------------------------------- etapa da cabeça
    @property
    def in_head_stage(self) -> bool:
        return self.head_start is not None and not self.collecting_done and not self.computing

    @property
    def head_total_s(self) -> float:
        return self.cfg.head_motion_settle_s + self.cfg.head_motion_step_s * len(HEAD_PROMPTS)

    @property
    def head_prompt(self) -> tuple[str, tuple[int, int]] | None:
        """Direção pedida agora (None fora da etapa ou durante o 'settle')."""
        if not self.in_head_stage:
            return None
        el = self._now - self.head_start - self.cfg.head_motion_settle_s
        if el < 0:
            return None
        i = min(int(el / self.cfg.head_motion_step_s), len(HEAD_PROMPTS) - 1)
        return HEAD_PROMPTS[i]

    def targets_done(self, now: float):
        if self.cfg.head_motion_step_s > 0:
            self.head_start = now
            self.index = len(self.targets) - 1  # mantém um alvo válido para quem consulta
        else:
            super().targets_done(now)

    def update(self, sample, now: float):
        self._now = now
        if self._run_pending(now):
            return
        if not self.in_head_stage:
            super().update(sample, now)
            return
        el = now - self.head_start
        if sample is not None and sample.valid:
            self._last_valid = now
            if el >= self.cfg.head_motion_settle_s:
                self.head_samples.append((sample.features(), sample.raw))
                self.head_observations.append(sample.obs)
        if el >= self.head_total_s and len(self.head_samples) >= self.min_samples:
            super().targets_done(now)

    @property
    def head_target(self) -> tuple[float, float]:
        return (self.screen_w / 2, self.screen_h / 2)

    def current_target(self) -> tuple[float, float]:
        return self.head_target if self.in_head_stage else super().current_target()

    def render(self, now: float) -> np.ndarray:
        self._now = now
        if self.computing or not self.in_head_stage:
            return super().render(now)
        canvas = np.full((self.screen_h, self.screen_w, 3), BG, dtype=np.uint8)
        cx, cy = (int(v) for v in self.head_target)
        cv2.circle(canvas, (cx, cy), 12, COLLECT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, (0, 0, 0), -1, cv2.LINE_AA)
        el = now - self.head_start
        k = min(1.0, el / self.head_total_s)
        cv2.ellipse(canvas, (cx, cy), (22, 22), -90, 0, 360 * k, COLLECT, 3, cv2.LINE_AA)

        prompt = self.head_prompt
        self._centered(canvas, "Mantenha o OLHAR no ponto verde e gire a CABECA devagar", 45)
        if prompt is None:
            self._centered(canvas, "Prepare-se...", 90)
        else:
            name, (dx, dy) = prompt
            if dx or dy:
                tip = (cx + dx * 160, cy + dy * 160)
                cv2.arrowedLine(canvas, (cx + dx * 45, cy + dy * 45), tip, TARGET, 6,
                                cv2.LINE_AA, tipLength=0.3)
                self._centered(canvas, f"Cabeca para a {name} (o quanto for confortavel)", 90)
            else:
                self._centered(canvas, "Volte a cabeca ao CENTRO", 90)
        self._centered(canvas, "Se nao puder mover a cabeca, apenas continue olhando o ponto",
                       self.screen_h - 75)
        if now - self._last_valid > 1.0:
            self._centered(canvas, "Rosto/olhos nao detectados - posicione-se em frente a camera",
                           135, WARN)
        return canvas

    # ------------------------------------------------------- ajuste
    def on_complete(self):
        data = CalibrationData()
        for i, t in enumerate(self.targets):
            feats, raws = self.arrays(i)
            data.grid.append((t, feats, raws))
            data.grid_obs.append(self.observations[i])
        data.head_obs = self.head_observations
        if self.head_samples:
            data.head_target = self.head_target
            data.head_features = np.array([f for f, _ in self.head_samples])
            data.head_raw = np.array([r for _, r in self.head_samples])

        try:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.raw_path = self.cfg.results_dir / f"calibracao_{stamp}.npz"
            data.save(self.raw_path)
        except OSError as exc:  # gravar os dados brutos não pode impedir a calibração
            self.raw_path = None
            print(f"[Calibração] Não foi possível salvar os dados brutos: {exc}")

        self.model, self.results, self.stats = fit_from_samples(
            data, (self.screen_w, self.screen_h),
            self.cfg.polynomial_degree, self.cfg.ridge_alpha,
            self.cfg.outlier_mad_threshold,
            head_weight_points=self.cfg.head_motion_weight,
        )
        errs = np.array([r.error_px for r in self.results])
        self.succeeded = True
        st = self.stats
        self.message_lines = [
            "Calibracao concluida",
            f"Erro de ajuste nos alvos: {errs.mean():.0f} px  (max {errs.max():.0f} px)",
        ]
        if "erro_loo_medio_px" in st:
            # Erro de cada alvo quando ele fica FORA do ajuste: estimativa
            # honesta do erro em pontos novos (o erro de ajuste é otimista).
            self.message_lines.append(
                f"Erro estimado em pontos novos: {st['erro_loo_medio_px']:.0f} px"
                f"  (iris: {st.get('fonte_iris', 'mediapipe')})")
        if "erro_mediano_cabeca_px" in st:
            self.message_lines.append(
                f"Erro com a cabeca em movimento: {st['erro_mediano_cabeca_px']:.0f} px (mediana)")
        self.message_lines.append("Branco = alvo, ponto colorido = estimativa. Use T para validar.")
        print(f"[Calibração] Concluída. Erro médio de ajuste {errs.mean():.1f} px, "
              f"máx {errs.max():.1f} px. Estatísticas: {st}")
        if self.raw_path is not None:
            print(f"[Calibração] Dados brutos em {self.raw_path}")

    def render_result(self, canvas):
        _draw_point_results(canvas, self.results, tol_px=0.05 * self.screen_w)
        super().render_result(canvas)


class ValidationProcedure(TargetProcedure):
    """Mede acurácia e precisão em alvos NÃO usados na calibração."""

    title = "Validacao"

    def __init__(self, screen_size, model: CalibrationModel, cfg: ValidationConfig,
                 calib_cfg: CalibrationConfig):
        targets = [(fx * screen_size[0], fy * screen_size[1]) for fx, fy in cfg.points]
        super().__init__(targets, screen_size, cfg.settle_s, cfg.collect_s,
                         calib_cfg.min_samples_per_point)
        self.cfg = cfg
        self.mad = calib_cfg.outlier_mad_threshold
        self.model = model
        self.results: list[PointResult] = []
        self.summary: dict = {}
        self.csv_path: Path | None = None

    def _px_to_deg(self, px: float) -> float | None:
        if not (self.cfg.screen_width_cm and self.cfg.viewing_distance_cm):
            return None
        cm = px * self.cfg.screen_width_cm / self.screen_w
        return math.degrees(math.atan2(cm, self.cfg.viewing_distance_cm))

    def on_complete(self):
        rows = []
        accuracies, precisions = [], []
        raw_data = CalibrationData()
        for i, target in enumerate(self.targets):
            feats, raws = self.arrays(i)
            raw_data.grid.append((target, feats, raws))
            raw_data.grid_obs.append(self.observations[i])
            keep = outlier_mask(feats, self.mad)
            s, raws = feats[keep], raws[keep]
            preds = self.model.predict(raws)
            med = np.median(preds, axis=0)
            pr = PointResult(target, (float(med[0]), float(med[1])), len(s))
            self.results.append(pr)
            accuracies.append(pr.error_px)
            # Precisão: dispersão (RMS) das estimativas em torno da média.
            precisions.append(float(np.sqrt(np.mean(np.sum((preds - preds.mean(0)) ** 2, 1)))))
            for p, f in zip(preds, s):
                rows.append([i, target[0], target[1], p[0], p[1], f[0], f[1],
                             float(np.hypot(p[0] - target[0], p[1] - target[1]))])

        acc = float(np.mean(accuracies))
        prec = float(np.mean(precisions))
        self.summary = {
            "data": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tela_px": [self.screen_w, self.screen_h],
            "n_alvos": len(self.targets),
            "acuracia_media_px": acc,
            "acuracia_mediana_px": float(np.median(accuracies)),
            "precisao_rms_media_px": prec,
            "acuracia_media_graus": self._px_to_deg(acc),
            "precisao_rms_media_graus": self._px_to_deg(prec),
            "por_alvo": [{"alvo": r.target, "estimativa": r.prediction,
                          "erro_px": r.error_px, "n": r.n_samples} for r in self.results],
        }

        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.cfg.results_dir / f"validacao_{stamp}.csv"
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alvo_idx", "alvo_x", "alvo_y", "est_x", "est_y",
                        "yaw_deg", "pitch_deg", "erro_px"])
            w.writerows(rows)
        # Medidas brutas: permitem testar outros modelos nesta mesma sessão
        # (tools/reavaliar.py).
        raw_data.save(self.cfg.results_dir / f"validacao_{stamp}.npz")
        (self.cfg.results_dir / f"validacao_{stamp}.json").write_text(
            json.dumps(self.summary, indent=2, ensure_ascii=False))

        self.succeeded = True
        deg = self.summary["acuracia_media_graus"]
        deg_s = f" (~{deg:.1f} graus)" if deg is not None else ""
        self.message_lines = [
            "Validacao concluida",
            f"Acuracia (erro medio): {acc:.0f} px{deg_s}",
            f"Precisao (RMS): {prec:.0f} px",
            f"Salvo em results/{self.csv_path.name}",
        ]
        print(f"[Validação] Acurácia {acc:.1f} px{deg_s}, precisão RMS {prec:.1f} px. "
              f"Resultados em {self.csv_path}")

    def render_result(self, canvas):
        _draw_point_results(canvas, self.results, tol_px=0.05 * self.screen_w)
        super().render_result(canvas)


class RecenterProcedure(TargetProcedure):
    """Correção rápida de deslocamento (1 ponto) sem refazer a calibração.

    O modelo já compensa rotação e translação da cabeça, mas aproximações
    (posição 3D do rosto estimada pelo MediaPipe, cansaço, óculos que
    escorregam) deixam um deslocamento residual, que é corrigido somando
    um offset à saída do modelo.
    """

    title = "Recentralizar"
    result_timeout_s = 1.5

    def __init__(self, screen_size, model: CalibrationModel, cfg: CalibrationConfig):
        center = (screen_size[0] / 2, screen_size[1] / 2)
        super().__init__([center], screen_size, cfg.settle_s, cfg.collect_s,
                         cfg.min_samples_per_point)
        self.model = model
        self.mad = cfg.outlier_mad_threshold

    def on_complete(self):
        feats, raws = self.arrays(0)
        pred = np.median(self.model.predict(raws[outlier_mask(feats, self.mad)]), axis=0)
        delta = np.array(self.targets[0]) - pred
        self.model.offset = self.model.offset + delta
        self.succeeded = True
        self.message_lines = [f"Recentralizado: deslocamento ({delta[0]:+.0f}, {delta[1]:+.0f}) px"]
        print(f"[Recentralizar] Offset ajustado em ({delta[0]:+.1f}, {delta[1]:+.1f}) px.")
