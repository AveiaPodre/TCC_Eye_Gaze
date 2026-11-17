"""
Procedimentos com alvos em tela cheia: calibração, validação e recentralização.

Cada procedimento é uma máquina de estados atualizada a cada frame pelo
loop principal (``update``) e desenhada numa imagem do tamanho da tela
(``render``). Assim a câmera e o rastreamento continuam rodando
normalmente durante o procedimento, sem loops aninhados.

Para cada alvo:
    1. "settle"  - o alvo aparece e encolhe (atrai o olhar); nada é coletado.
       Depois de ``settle_s``, espera o olhar ESTABILIZAR (fixação) antes de
       coletar, com limite de ``settle_max_s``. Com tempo fixo, as primeiras
       amostras às vezes ainda eram da sacada ou de um olhar "procurando".
    2. "collect" - amostras válidas (rosto visível, sem piscada) são guardadas.
       Se o rosto sumir, o alvo espera até juntar o mínimo de amostras.

Cada amostra guarda duas coisas: os ângulos (yaw, pitch) do vetor do olhar,
usados para rejeitar outliers, detectar a fixação e no CSV, e o vetor bruto
``raw`` (âncoras e pupilas em cm, pose da cabeça), entrada da calibração
(ver gaze.py e calibration.py).

Calibração por perseguição (opcional, ``CalibrationConfig.pursuit``): um
ponto percorre a tela devagar e a pessoa o segue com os olhos (Pfeuffer et
al., 2013). Cada quadro vira uma amostra com o seu próprio alvo (a posição do
ponto ~0,1 s antes, o atraso do olho), cobrindo a tela de forma contínua em
~13-21 s. "soma": grade -> perseguição -> etapa 2; "alternativa":
perseguição + 5 alvos fixos (no lugar da grade).

A calibração percorre os alvos em ordem aleatória. Depois do ajuste, alvos
com erro muito alto são coletados de novo, uma vez. Em seguida, o app roda
a conferência (``RecenterProcedure``, 4 pontos), que corrige o desvio
constante do começo do uso.

O teste da cabeça (``HeadStageProcedure``, tecla K ou ``--cabeca``) mede o
erro com a cabeça em poses novas: o alvo fica no centro e uma seta pede que
a pessoa gire a cabeça sem tirar o olho do alvo. As amostras não entram no
ajuste.
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
                          fit_from_samples, head_angles, head_test_error, outlier_mask)
from .config import CalibrationConfig, ValidationConfig
from .gaze import RAW_FACE, RAW_SIZE, GazeSample

BG = (40, 40, 40)
WHITE = (240, 240, 240)
GRAY = (150, 150, 150)
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

    def __init__(self, targets, screen_size, settle_s, collect_s, min_samples,
                 settle_max_s: float | None = None, fixation_window: int = 6,
                 fixation_max_deg: float = 0.8, order=None):
        self.targets = [tuple(map(float, t)) for t in targets]
        self.screen_w, self.screen_h = screen_size
        self.settle_s = settle_s
        self.settle_max_s = settle_s if settle_max_s is None else max(settle_max_s, settle_s)
        self.fixation_window = fixation_window
        self.fixation_max_deg = fixation_max_deg
        self.collect_s = collect_s
        self.min_samples = min_samples
        # Por alvo: lista de (features (2,), raw (RAW_SIZE,)).
        self.samples: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in self.targets]
        # Landmarks e instante (s, relógio do app) de cada amostra.
        self.observations: list[list] = [[] for _ in self.targets]
        self.times: list[list[float]] = [[] for _ in self.targets]
        # Quadros sem amostra válida (rosto perdido, piscada) durante a coleta
        # de cada alvo: perda de dados.
        self.invalid: list[int] = [0 for _ in self.targets]
        # Ordem de visita (índices em ``targets``) e posição atual nela.
        self.order: list[int] = list(range(len(self.targets))) if order is None else list(order)
        self.pos = 0
        self._phase_start: float | None = None
        self._collect_start: float | None = None
        self._recent: list[np.ndarray] = []   # features recentes (detecção de fixação)
        self._last_valid = 0.0
        self.collecting_done = False   # todos os alvos coletados
        self.finished = False          # usuário já viu o resultado
        self.succeeded = False
        self.message_lines: list[str] = []
        self._result_since: float | None = None
        # Coleta terminou; o ajuste roda no próximo frame (depois de a tela
        # "Calculando..." ter sido exibida).
        self.computing = False
        # on_complete pode pedir para voltar a coletar (repetição de alvos).
        self.restart_collection = False
        # Modo manual: o alvo só avança quando o usuário confirma (tecla) que
        # está olhando para ele; as amostras são as de antes da confirmação.
        self.manual = False
        self.manual_window_s = 1.0      # amostras usadas: esta janela antes da tecla...
        self.manual_reaction_s = 0.15   # ...descontado o tempo de reação (mão/tecla)
        self.manual_min_s = 0.5         # confirmações antes disto são ignoradas
        self._manual_buf: list = []     # (t, features, raw, obs) do alvo atual
        self._manual_msg: tuple[str, float] | None = None

    def _blank(self) -> np.ndarray:
        """Fundo da tela de alvos. Copiar um fundo pronto custa bem menos que
        criar a imagem 1920x1080 a cada quadro (~10 ms), tempo que atrasava a
        leitura da câmera durante os procedimentos."""
        bg = getattr(self, "_bg", None)
        if bg is None or bg.shape[:2] != (self.screen_h, self.screen_w):
            bg = self._bg = np.full((self.screen_h, self.screen_w, 3), BG, dtype=np.uint8)
        return bg.copy()

    @property
    def index(self) -> int:
        """Índice (em ``targets``) do alvo atual."""
        return self.order[min(self.pos, len(self.order) - 1)]

    # ------------------------------------------------------------ estado
    def _run_pending(self, now: float) -> bool:
        if not self.computing:
            return False
        self.on_complete()
        self.computing = False
        if self.restart_collection:
            self.restart_collection = False
            self._start_phase(None)
            return True
        self.collecting_done = True
        self._result_since = now
        return True

    def _start_phase(self, now: float | None):
        self._phase_start = now
        self._collect_start = None
        self._recent = []
        self._manual_buf = []

    def _is_fixating(self) -> bool:
        n = self.fixation_window
        if len(self._recent) < n:
            return False
        win = np.asarray(self._recent[-n:])
        med = np.median(win, axis=0)
        return float(np.max(np.hypot(*(win - med).T))) < self.fixation_max_deg

    @property
    def collecting(self) -> bool:
        """True quando o alvo atual já está na fase de coleta."""
        return self._collect_start is not None

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
            self._start_phase(now)
            self._last_valid = now
        elapsed = now - self._phase_start
        valid = sample is not None and sample.valid
        if valid:
            self._last_valid = now

        if self.manual:
            if valid:
                self._manual_buf.append((now, sample.features(), sample.raw, sample.obs))
                horizon = now - 3.0 * self.manual_window_s
                while self._manual_buf and self._manual_buf[0][0] < horizon:
                    self._manual_buf.pop(0)
            return

        if self._collect_start is None:
            if valid:
                self._recent.append(sample.features())
                del self._recent[:-self.fixation_window]
            if elapsed >= self.settle_s and (self._is_fixating() or elapsed >= self.settle_max_s):
                self._collect_start = now

        if self._collect_start is not None:
            if valid:
                self.samples[self.index].append((sample.features(), sample.raw))
                self.observations[self.index].append(sample.obs)
                self.times[self.index].append(now)
            else:
                self.invalid[self.index] += 1
            enough = len(self.samples[self.index]) >= self.min_samples
            if now - self._collect_start >= self.collect_s and enough:
                self.pos += 1
                self._start_phase(now)
                if self.pos >= len(self.order):
                    self.targets_done(now)

    def targets_done(self, now: float):
        """Todos os alvos coletados (subclasses podem acrescentar etapas)."""
        self.computing = True

    @property
    def awaiting_confirmation(self) -> bool:
        """Modo manual, num alvo da grade, esperando a tecla de confirmação."""
        return (self.manual and not self.collecting_done and not self.computing
                and self.pos < len(self.order))

    def confirm(self, now: float) -> bool:
        """Modo manual: o usuário confirma que está olhando para o alvo.

        Usa as amostras da janela ``manual_window_s`` ANTES da tecla (sem os
        últimos ``manual_reaction_s``): ao apertar a tecla a pessoa costuma
        piscar ou olhar para o teclado, então as amostras de depois não
        servem. Recusa (e avisa) se o olhar não estava parado nessa janela.
        """
        if not self.awaiting_confirmation or self._phase_start is None:
            return False
        if now - self._phase_start < self.manual_min_s:
            return False
        t1 = now - self.manual_reaction_s
        win = [b for b in self._manual_buf if t1 - self.manual_window_s <= b[0] <= t1]
        if len(win) < self.min_samples:
            self._manual_msg = ("Rosto/olhos nao detectados: olhe para o ponto e tente de novo", now)
            return False
        feats = np.array([b[1] for b in win])
        med = np.median(feats, axis=0)
        if float(np.percentile(np.hypot(*(feats - med).T), 80)) > 2.0 * self.fixation_max_deg:
            self._manual_msg = ("Olhar ainda se movendo: fixe o ponto e aperte de novo", now)
            return False
        for t, f, r, o in win:
            self.samples[self.index].append((f, r))
            self.observations[self.index].append(o)
            self.times[self.index].append(t)
        self._manual_msg = None
        self.pos += 1
        self._start_phase(now)
        if self.pos >= len(self.order):
            self.targets_done(now)
        return True

    def arrays(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(features (n, 2), raw (n, RAW_SIZE)) do alvo i."""
        feats = np.array([f for f, _ in self.samples[i]]).reshape(-1, 2)
        raws = np.array([r for _, r in self.samples[i]])
        return feats, raws

    def current_target(self) -> tuple[float, float]:
        return self.targets[self.index]

    def acknowledge(self):
        """Chamado quando o usuário aperta uma tecla na tela de resultado."""
        if self.collecting_done:
            self.finished = True

    def on_complete(self):  # sobrescrito nas subclasses
        self.succeeded = True

    # ----------------------------------------------------------- desenho
    def status_line(self) -> str:
        return (f"{self.title}: olhe para o ponto ({self.pos + 1}/{len(self.order)})"
                "  -  ESC cancela")

    def render(self, now: float) -> np.ndarray:
        canvas = self._blank()
        if self.computing:
            self._centered(canvas, f"{self.title}: calculando...", self.screen_h // 2)
            return canvas
        if self.collecting_done:
            self.render_result(canvas)
            _text(canvas, "Pressione qualquer tecla para continuar",
                  (30, self.screen_h - 30), 0.7)
            return canvas

        tx, ty = (int(round(v)) for v in self.current_target())
        elapsed = 0.0 if self._phase_start is None else now - self._phase_start

        if self.manual:
            cv2.circle(canvas, (tx, ty), 16, TARGET, 3, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 4, WHITE, -1, cv2.LINE_AA)
        elif self._collect_start is None:
            # Alvo encolhendo enquanto o olho chega; depois, um anel pequeno
            # até o olhar estabilizar.
            k = min(1.0, elapsed / max(self.settle_s, 1e-6))
            r = int(40 - 28 * k)
            cv2.circle(canvas, (tx, ty), r, TARGET, 3, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 4, WHITE, -1, cv2.LINE_AA)
        else:
            n = len(self.samples[self.index])
            k = min(1.0, (now - self._collect_start) / max(self.collect_s, 1e-6))
            k = min(k, n / self.min_samples) if n < self.min_samples else k
            cv2.circle(canvas, (tx, ty), 12, COLLECT, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (tx, ty), (22, 22), -90, 0, 360 * k, COLLECT, 3, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 3, (0, 0, 0), -1, cv2.LINE_AA)

        # Texto no lado oposto ao alvo, para não competir com ele pelo olhar.
        top = ty >= self.screen_h / 2
        y0 = 45 if top else self.screen_h - 75
        self._centered(canvas, self.status_line(), y0)
        if self.manual:
            msg = self._manual_msg
            if msg is not None and now - msg[1] < 2.5:
                self._centered(canvas, msg[0], y0 + 40, WARN)
            else:
                self._centered(canvas, "Olhe para o ponto e aperte ESPACO (ou ENTER)", y0 + 40)
        if now - self._last_valid > 1.0:
            self._centered(canvas, "Rosto/olhos nao detectados - posicione-se em frente a camera",
                           y0 + 40, WARN)
        return canvas

    def _centered(self, canvas, s, y, color=WHITE, scale=0.8):
        (tw, _), _ = cv2.getTextSize(s, FONT, scale, 2)
        _text(canvas, s, ((self.screen_w - tw) // 2, y), scale, color)

    def render_result(self, canvas: np.ndarray):
        """Escreve as mensagens num bloco centralizado que não cubra nenhum alvo."""
        if not self.message_lines:
            return
        margin = 35
        best = None   # (alvos cobertos, -escala, y0, x0, escala, passo)
        for scale in (0.8, 0.7, 0.6):
            step = int(50 * scale)
            widths = [cv2.getTextSize(l, FONT, scale, 2)[0][0] for l in self.message_lines]
            bw, bh = max(widths) + 20, step * len(self.message_lines) + 10
            x0 = max((self.screen_w - bw) // 2, 0)
            for cy in range(10, max(self.screen_h - 60 - bh, 11), 10):
                covered = sum(x0 - margin < tx < x0 + bw + margin
                              and cy - margin < ty < cy + bh + margin for tx, ty in self.targets)
                key = (covered, -scale)
                if best is None or key < best[:2]:
                    best = (covered, -scale, cy, x0, scale, step)
            if best[0] == 0:
                break   # cabe sem cobrir alvos nesta escala
        _, _, y0, x0, scale, step = best
        for i, line in enumerate(self.message_lines):
            _text(canvas, line, (x0 + 10, y0 + int(0.75 * step) + step * i), scale)


def _draw_point_results(canvas, results: list[PointResult], tol_px: float):
    for pr in results:
        t = tuple(int(round(v)) for v in pr.target)
        p = tuple(int(round(v)) for v in pr.prediction)
        color = COLLECT if pr.error_px <= tol_px else WARN
        cv2.circle(canvas, t, 14, WHITE, 2, cv2.LINE_AA)
        cv2.line(canvas, t, p, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p, 7, color, -1, cv2.LINE_AA)
        _text(canvas, f"{pr.error_px:.0f}px", (t[0] + 18, t[1] - 18), 0.6)


def shrink_offset(errors: np.ndarray, c: float = 4.0) -> tuple[np.ndarray, float]:
    """Desvio constante a corrigir a partir dos erros (n, 2) em n alvos:
    a MÉDIA, encolhida pela incerteza dela (James-Stein):
    f = |m|² / (|m|² + c·se²), com se² = soma das variâncias / n.

    Nas 48 validações gravadas (os 4 primeiros alvos estimam o desvio, os 5
    seguintes medem o erro), corrigir pela média inteira levava o erro de 105
    a 90 px, mas piorava 31% das sessões; com o fator adaptativo (c = 4), 84
    px, piorando 19%. Desvios grandes e coerentes (f ~ 1) continuam
    corrigidos quase por inteiro; um fator fixo de 0,5 deixava metade deles
    (123 contra 91 px com um desvio artificial de 180 px). Devolve (desvio
    encolhido, f)."""
    E = np.asarray(errors, dtype=float)
    m = E.mean(axis=0)
    if len(E) < 2:
        return m, 1.0
    se2 = float(np.sum(np.var(E, axis=0, ddof=1))) / len(E)
    mm = float(np.dot(m, m))
    f = mm / (mm + c * se2) if mm + c * se2 > 0 else 0.0
    return f * m, f


def quality_label(error_px: float, screen_w: int, good_frac: float = 0.047) -> tuple[str, str]:
    """Classificação do erro estimado em pontos novos (fração da largura da tela).
    ``good_frac``: limite de "boa" (CalibrationConfig.redo_loo_frac)."""
    frac = error_px / screen_w
    if frac <= good_frac:
        return "boa", "pode usar alvos medios (~10% da tela)"
    if frac <= 0.08:
        return "razoavel", "use alvos grandes; R corrige desvios"
    return "ruim", "refaca: rosto bem iluminado, cabeca parada nos alvos"


# Direções da etapa da cabeça: (nome, direção da seta na tela). yaw > 0 é a
# direita do usuário; dy = -1 (seta para cima) é pitch > 0.
HEAD_PROMPTS = (
    ("DIREITA", (1, 0)),
    ("ESQUERDA", (-1, 0)),
    ("CIMA", (0, -1)),
    ("BAIXO", (0, 1)),
    ("CENTRO", (0, 0)),
)


def head_progress(delta: np.ndarray, direction: tuple[int, int],
                  target_deg: tuple[float, float]) -> float:
    """Fração do giro pedido já alcançada na direção da seta (pode passar de 1)."""
    dx, dy = direction
    if dx:
        return float(delta[0] * dx / target_deg[0])
    if dy:
        return float(delta[1] * -dy / target_deg[1])
    return 0.0


def pursuit_path(screen_size, mode: str, margin: float) -> np.ndarray:
    """Vértices (n, 2) do caminho do ponto na perseguição. "soma": o
    perímetro (as bordas, onde o erro é maior); "alternativa": o perímetro e
    as linhas do meio (horizontal e vertical), que cobrem o interior."""
    w, h = screen_size
    x0, x1, y0, y1 = margin * w, (1 - margin) * w, margin * h, (1 - margin) * h
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    if mode == "alternativa":
        pts += [(x0, h / 2), (x1, h / 2), (w / 2, y1), (w / 2, y0)]
    return np.array(pts, dtype=float)


def _along(path: np.ndarray, cum: np.ndarray, dist: float) -> tuple[np.ndarray, int]:
    """Ponto a ``dist`` px do início do caminho e o índice do segmento."""
    dist = float(np.clip(dist, 0.0, cum[-1]))
    seg = int(min(np.searchsorted(cum, dist, side="right") - 1, len(path) - 2))
    a, b = path[seg], path[seg + 1]
    length = max(cum[seg + 1] - cum[seg], 1e-9)
    return a + (b - a) * ((dist - cum[seg]) / length), seg


class CalibrationProcedure(TargetProcedure):
    title = "Calibracao"

    def __init__(self, screen_size, cfg: CalibrationConfig):
        mode = getattr(cfg, "pursuit", "nenhuma")
        if mode == "alternativa":
            # Perseguição no lugar da grade, mais poucos alvos fixos (o
            # critério de aceitação e a repetição de alvos usam só eles).
            targets = [(fx * screen_size[0], fy * screen_size[1])
                       for fx, fy in cfg.pursuit_points]
        else:
            targets = build_grid(*screen_size, cfg.grid_cols, cfg.grid_rows, cfg.margin_ratio)
        rng = np.random.default_rng(cfg.random_seed)
        order = list(range(len(targets)))
        if cfg.shuffle_targets:
            rng.shuffle(order)
        # Etapa 2 (refinamento): alvos internos depois da grade, também em
        # ordem aleatória; o ajuste usa todos os pontos juntos.
        self.n_grid = len(targets)
        refine = [] if mode == "alternativa" else [
            (fx * screen_size[0], fy * screen_size[1]) for fx, fy in getattr(cfg, "refine_points", ())]
        stage2 = list(range(len(targets), len(targets) + len(refine)))
        if cfg.shuffle_targets:
            rng.shuffle(stage2)
        targets = targets + refine
        order = order + stage2
        super().__init__(targets, screen_size, cfg.settle_s, cfg.collect_s,
                         cfg.min_samples_per_point, settle_max_s=cfg.settle_max_s,
                         fixation_window=cfg.fixation_window,
                         fixation_max_deg=cfg.fixation_max_deg, order=order)
        self.cfg = cfg
        self.manual = bool(getattr(cfg, "manual", False))
        self.model: CalibrationModel | None = None
        self.raw_path: Path | None = None
        self.results: list[PointResult] = []
        self.stats: dict = {}
        # Repetição de alvos ruins.
        self.retried_targets: list[int] = []
        # Perseguição: roda quando ``pos`` chega a ``pursuit_at`` (depois da
        # grade em "soma"; antes dos alvos fixos em "alternativa").
        self.pursuit_mode = mode
        self.pursuit_done = mode not in ("soma", "alternativa")
        self.pursuit_at = 0 if mode == "alternativa" else self.n_grid
        self.pursuit_path = (None if self.pursuit_done
                             else pursuit_path(screen_size, mode, cfg.margin_ratio))
        self._p_cum = (None if self.pursuit_path is None else np.concatenate(
            [[0.0], np.cumsum(np.hypot(*np.diff(self.pursuit_path, axis=0).T))]))
        self._p_start: float | None = None
        self._p_pos: np.ndarray | None = None
        self._p_last_feat: np.ndarray | None = None
        self.p_targets, self.p_feats, self.p_raw, self.p_obs, self.p_times = [], [], [], [], []
        self.p_invalid = 0

    # ------------------------------------------------------- perseguição
    @property
    def pursuing(self) -> bool:
        """A perseguição é a etapa atual."""
        return (not self.pursuit_done and not self.computing and not self.collecting_done
                and self.pos >= self.pursuit_at)

    @property
    def pursuit_duration_s(self) -> float:
        if self._p_cum is None:
            return 0.0
        return self.cfg.pursuit_hold_s + self._p_cum[-1] / self.cfg.pursuit_speed_px_s

    def update(self, sample: GazeSample | None, now: float):
        if self.pursuing:
            self._pursuit_update(sample, now)
            return
        super().update(sample, now)

    def targets_done(self, now: float):
        if not self.pursuit_done:
            return          # a perseguição ainda vem; o ajuste roda depois dela
        super().targets_done(now)

    def _pursuit_update(self, sample: GazeSample | None, now: float):
        c = self.cfg
        if self._p_start is None:
            self._p_start = now
            self._last_valid = now
        valid = sample is not None and sample.valid
        if valid:
            self._last_valid = now
        speed = c.pursuit_speed_px_s
        t = now - self._p_start - c.pursuit_hold_s     # tempo em movimento
        if t < 0:
            self._p_pos = self.pursuit_path[0].copy()  # parado no início (o olho chega)
            return
        if t * speed >= self._p_cum[-1]:
            self.pursuit_done = True
            self._p_pos = None
            self._start_phase(now)
            if self.pos >= len(self.order):
                self.computing = True
            return
        self._p_pos, seg = _along(self.pursuit_path, self._p_cum, t * speed)
        if not valid:
            self.p_invalid += 1
            self._p_last_feat = None
            return
        f = sample.features()
        jump = (self._p_last_feat is not None
                and float(np.hypot(*(f - self._p_last_feat))) > c.pursuit_jump_deg)
        self._p_last_feat = f
        # Descarta o começo do movimento (o olho ainda acelera), o trecho
        # logo depois de cada canto (o olho corta caminho e corrige com uma
        # sacada) e saltos entre quadros (sacadas, olhar perdido).
        since_corner = t - self._p_cum[seg] / speed
        td = t - c.pursuit_delay_s
        if (t < c.pursuit_skip_s or td < 0 or jump
                or (seg > 0 and since_corner < c.pursuit_corner_skip_s)):
            return
        # O olho segue o ponto com um pequeno atraso: o alvo da amostra é a
        # posição do ponto ``pursuit_delay_s`` antes.
        target, _ = _along(self.pursuit_path, self._p_cum, td * speed)
        self.p_targets.append(target)
        self.p_feats.append(f)
        self.p_raw.append(sample.raw)
        self.p_obs.append(sample.obs)
        self.p_times.append(now)

    def current_target(self) -> tuple[float, float]:
        if self.pursuing:
            p = self.pursuit_path[0] if self._p_pos is None else self._p_pos
            return float(p[0]), float(p[1])
        return super().current_target()

    @property
    def awaiting_confirmation(self) -> bool:
        return super().awaiting_confirmation and not self.pursuing

    def render(self, now: float) -> np.ndarray:
        if not self.pursuing:
            return super().render(now)
        canvas = self._blank()
        tx, ty = (int(round(v)) for v in self.current_target())
        el = 0.0 if self._p_start is None else now - self._p_start
        if el < self.cfg.pursuit_hold_s:
            k = min(1.0, el / max(self.cfg.pursuit_hold_s, 1e-6))
            cv2.circle(canvas, (tx, ty), int(40 - 26 * k), TARGET, 3, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 4, WHITE, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (tx, ty), 11, COLLECT, -1, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 3, (0, 0, 0), -1, cv2.LINE_AA)
        y0 = int(0.3 * self.screen_h)
        self._centered(canvas, self.status_line(), y0)
        self._centered(canvas, "Siga o ponto so com os olhos, sem mexer a cabeca", y0 + 40)
        if now - self._last_valid > 1.0:
            self._centered(canvas, "Rosto/olhos nao detectados - posicione-se em frente a camera",
                           y0 + 80, WARN)
        return canvas

    @property
    def redo_recommended(self) -> bool:
        """Calibração concluída, mas com erro em pontos novos acima do critério
        de aceitação (CalibrationConfig.redo_loo_frac)."""
        loo = self.stats.get("erro_loo_medio_px")
        return (self.succeeded and loo is not None
                and loo > getattr(self.cfg, "redo_loo_frac", 0.047) * self.screen_w)

    @property
    def retrying(self) -> bool:
        return bool(self.retried_targets) and not self.collecting_done

    def status_line(self) -> str:
        n = len(self.order)
        if self.pursuing:
            return "Calibracao: siga o ponto com os olhos  -  ESC cancela"
        if self.retrying:
            return f"Calibracao: repetindo alvos com erro alto ({self.pos + 1}/{n})  -  ESC cancela"
        if self.index >= self.n_grid:
            return (f"Calibracao, etapa 2 (refinamento): olhe para o ponto ({self.pos + 1}/{n})"
                    "  -  ESC cancela")
        return super().status_line()

    # ------------------------------------------------------- ajuste
    def _collected_data(self) -> CalibrationData:
        data = CalibrationData()
        for i, t in enumerate(self.targets):
            feats, raws = self.arrays(i)
            data.grid.append((t, feats, raws))
            data.grid_obs.append(self.observations[i])
            data.grid_times.append(np.asarray(self.times[i], dtype=float))
        if self.p_raw:
            data.pursuit_targets = np.array(self.p_targets, dtype=float)
            data.pursuit_features = np.array(self.p_feats, dtype=float).reshape(-1, 2)
            data.pursuit_raw = np.array(self.p_raw, dtype=float)
            data.pursuit_times = np.array(self.p_times, dtype=float)
            data.pursuit_obs = list(self.p_obs)
        return data

    def _fit(self, data: CalibrationData):
        return fit_from_samples(
            data, (self.screen_w, self.screen_h), self.cfg.screen_width_cm,
            self.cfg.outlier_mad_threshold,
            head_max_deg=self.cfg.head_max_deg,
            max_degree=self.cfg.polynomial_degree,
            assumed_width_cm=self.cfg.assumed_screen_width_cm,
            pursuit_weight=getattr(self.cfg, "pursuit_weight", 6.0),
        )

    def bad_targets(self, stats: dict) -> list[int]:
        """Alvos a repetir: erro LOO muito acima da mediana (pior primeiro)."""
        errs = np.asarray(stats.get("erro_loo_por_alvo_px", []), dtype=float)
        ids = stats.get("erro_loo_alvos", [])
        if len(errs) < 5 or len(ids) != len(errs):
            return []
        limit = max(self.cfg.retry_ratio * float(np.median(errs)),
                    self.cfg.retry_min_frac * self.screen_w)
        bad = [(e, i) for e, i in zip(errs, ids) if e > limit]
        bad.sort(reverse=True)
        return [int(i) for _, i in bad[:self.cfg.retry_max_targets]]

    def on_complete(self):
        data = self._collected_data()
        self.model, self.results, self.stats = self._fit(data)

        if self.cfg.retry_bad_targets and not self.retried_targets:
            bad = self.bad_targets(self.stats)
            if bad:
                print(f"[Calibração] Repetindo {len(bad)} alvo(s) com erro alto: "
                      + ", ".join(f"{self.targets[i][0]:.0f},{self.targets[i][1]:.0f}"
                                  for i in bad))
                self.retried_targets = bad
                for i in bad:
                    self.samples[i] = []
                    self.observations[i] = []
                    self.times[i] = []
                self.order = bad
                self.pos = 0
                self.restart_collection = True
                return

        try:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.raw_path = self.cfg.results_dir / f"calibracao_{stamp}.npz"
            data.save(self.raw_path)
        except OSError as exc:  # gravar os dados brutos não pode impedir a calibração
            self.raw_path = None
            print(f"[Calibração] Não foi possível salvar os dados brutos: {exc}")

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
            label, advice = quality_label(st["erro_loo_medio_px"], self.screen_w,
                                          getattr(self.cfg, "redo_loo_frac", 0.047))
            self.message_lines.append(
                f"Erro estimado em pontos novos: {st['erro_loo_medio_px']:.0f} px"
                f"  (pupila {st.get('fonte_pupila', 'mediapipe')},"
                f" correcao virtual->real: {st.get('correcao', '?')})")
            if "erro_loo_monitor_virtual_px" in st:
                self.message_lines.append(
                    "So o monitor virtual (sem correcao): "
                    f"{st['erro_loo_monitor_virtual_px']:.0f} px")
            self.message_lines.append(f"Qualidade: {label} - {advice}")
        if "erro_cabeca_px" in st:
            self.message_lines.append(
                f"Teste com a cabeca em poses novas: {st['erro_cabeca_px']:.0f} px "
                f"({st['n_cabeca_usadas']} de {st['n_cabeca_total']} amostras)")
        if "n_perseguicao" in st:
            self.message_lines.append(
                f"Perseguicao: {st['n_perseguicao']} amostras, erro de ajuste "
                f"{st['erro_ajuste_perseguicao_px']:.0f} px")
        elif self.pursuit_mode in ("soma", "alternativa"):
            self.message_lines.append(
                f"Perseguicao: so {len(self.p_raw)} amostras validas, nao usada no ajuste")
        if self.retried_targets:
            self.message_lines.append(f"{len(self.retried_targets)} alvo(s) repetido(s)")
        self.message_lines.append("Branco = alvo, ponto colorido = estimativa.")
        if self.redo_recommended:
            self.message_lines.append(
                "RECOMENDADO REFAZER: ESPACO refaz agora (sente-se como vai usar, rosto "
                "iluminado, olhe o centro de cada ponto); outra tecla aceita assim.")
            self.result_timeout_s = 30.0      # tempo para decidir
        print(f"[Calibração] Concluída. Erro médio de ajuste {errs.mean():.1f} px, "
              f"máx {errs.max():.1f} px. Estatísticas: "
              f"{ {k: v for k, v in st.items() if k not in ('candidatos', 'erro_loo_por_alvo_px', 'erro_loo_alvos')} }")
        if self.raw_path is not None:
            print(f"[Calibração] Dados brutos em {self.raw_path}")

    def render_result(self, canvas):
        _draw_point_results(canvas, self.results, tol_px=0.05 * self.screen_w)
        super().render_result(canvas)


class HeadStageProcedure(TargetProcedure):
    """Teste da cabeça (tecla K ou ``--cabeca``): olhar fixo no centro, cabeça girando
    para a direita, a esquerda, para cima e para baixo, seguindo setas.

    As amostras NÃO entram no ajuste: medem o erro com a cabeça em poses
    novas (``calibration.head_test_error``), que fica em ``model.info`` e é
    gravado em ``results/cabeca_<data>.npz``. Roda depois da validação
    imediata: logo depois da grade, o efeito passageiro dos giros (~200 px
    com a cabeça já de volta) contaminava a validação (metodologia, 5.11).
    """

    title = "Teste da cabeca"

    def __init__(self, screen_size, model: CalibrationModel, cfg: CalibrationConfig):
        center = (screen_size[0] / 2, screen_size[1] / 2)
        super().__init__([center], screen_size, cfg.settle_s, cfg.collect_s,
                         cfg.min_samples_per_point)
        self.cfg = cfg
        self.model = model
        self.head_start: float | None = None
        self.head_step = 0
        self._step_start: float | None = None
        self._reached_since: float | None = None
        self.head_base = np.asarray(model.head_ref, dtype=float)   # pose da grade
        self.head_delta: np.ndarray | None = None     # desvio atual em relação a ela
        self.head_samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.head_observations: list = []
        self.stats: dict = {}
        self.raw_path: Path | None = None

    @property
    def in_head_stage(self) -> bool:
        return self.head_start is not None and not self.collecting_done and not self.computing

    @property
    def head_prompt(self) -> tuple[str, tuple[int, int]] | None:
        """Direção pedida agora (None fora da etapa ou durante o 'prepare-se')."""
        if not self.in_head_stage or self._step_start is None:
            return None
        return HEAD_PROMPTS[min(self.head_step, len(HEAD_PROMPTS) - 1)]

    @property
    def head_target(self) -> tuple[float, float]:
        return (self.screen_w / 2, self.screen_h / 2)

    def update(self, sample, now: float):
        if self._run_pending(now):
            return
        if self.collecting_done:
            if self._result_since is None:
                self._result_since = now
            elif now - self._result_since > self.result_timeout_s:
                self.finished = True
            return
        if self.head_start is None:
            self.head_start = now
            self._last_valid = now
        self._update_head(sample, now)

    def _update_head(self, sample, now: float):
        el = now - self.head_start
        valid = sample is not None and sample.valid
        if valid:
            self._last_valid = now
            self.head_delta = head_angles(sample.raw)[0] - self.head_base
        if el < self.cfg.head_motion_settle_s:
            return
        if self._step_start is None:
            self._step_start = now
        if valid:
            self.head_samples.append((sample.features(), sample.raw))
            self.head_observations.append(sample.obs)

        name, direction = HEAD_PROMPTS[self.head_step]
        step_el = now - self._step_start
        if direction == (0, 0):
            # Etapa final: cabeça de volta ao centro.
            if step_el >= self.cfg.head_center_s and len(self.head_samples) >= self.min_samples:
                self.computing = True     # resultado no próximo quadro
            return
        reached = (self.head_delta is not None and
                   head_progress(self.head_delta, direction, self.cfg.head_target_deg)
                   >= self.cfg.head_reach_frac)
        if reached and self._reached_since is None:
            self._reached_since = now
        held = self._reached_since is not None and now - self._reached_since >= self.cfg.head_hold_s
        if held or step_el >= self.cfg.head_step_timeout_s:
            self.head_step += 1
            self._step_start = now
            self._reached_since = None

    def on_complete(self):
        if len(self.head_samples) < 5:
            self.message_lines = ["Teste da cabeca: amostras insuficientes"]
            return
        data = CalibrationData(head_target=self.head_target,
                               head_features=np.array([f for f, _ in self.head_samples]),
                               head_raw=np.array([r for _, r in self.head_samples]),
                               head_obs=self.head_observations)
        self.stats = head_test_error(self.model, data.head_raw, data.head_target,
                                     max_deg=self.cfg.head_max_deg, base=self.head_base)
        self.model.info.update(self.stats)
        try:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.raw_path = self.cfg.results_dir / f"cabeca_{stamp}.npz"
            data.save(self.raw_path)
        except OSError as exc:
            self.raw_path = None
            print(f"[Teste da cabeça] Não foi possível salvar os dados brutos: {exc}")
        self.succeeded = True
        st = self.stats
        if "erro_cabeca_px" in st:
            by_dir = "  ".join(f"{k} {v:.0f}" for k, v in st["erro_cabeca_por_direcao_px"].items())
            self.message_lines = [
                f"Teste da cabeca: {st['erro_cabeca_px']:.0f} px com a cabeca em poses novas",
                f"Por direcao (px): {by_dir}",
                f"{st['n_cabeca_usadas']} de {st['n_cabeca_total']} amostras dentro da faixa util"]
        else:
            self.message_lines = ["Teste da cabeca: a cabeca quase nao girou; sem medida"]
        print(f"[Teste da cabeça] {st}")

    def render(self, now: float) -> np.ndarray:
        if self.computing or self.collecting_done:
            return super().render(now)
        canvas = self._blank()
        cx, cy = (int(v) for v in self.head_target)
        cv2.circle(canvas, (cx, cy), 12, COLLECT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, (0, 0, 0), -1, cv2.LINE_AA)

        self._centered(canvas, "Mantenha o OLHAR no ponto verde e gire so a CABECA", 45)
        prompt = self.head_prompt
        if prompt is None:
            self._centered(canvas, "Prepare-se...", 90)
        else:
            name, (dx, dy) = prompt
            if dx or dy:
                self._draw_head_arrow(canvas, (cx, cy), (dx, dy))
                self._centered(canvas, f"Cabeca para a {name} ate a seta ficar verde", 90)
            else:
                self._centered(canvas, "Volte a cabeca ao CENTRO", 90)
        self._centered(canvas, "Se nao puder mover a cabeca, apenas continue olhando o ponto",
                       self.screen_h - 75)
        if now - self._last_valid > 1.0:
            self._centered(canvas, "Rosto/olhos nao detectados - posicione-se em frente a camera",
                           135, WARN)
        return canvas

    def _draw_head_arrow(self, canvas, center, direction):
        """Seta que enche conforme a cabeça gira: laranja -> verde ao chegar
        no ângulo pedido; vermelha se passar da faixa aproveitada."""
        cx, cy = center
        dx, dy = direction
        base = (cx + dx * 45, cy + dy * 45)
        tip = (cx + dx * 220, cy + dy * 220)
        cv2.arrowedLine(canvas, base, tip, GRAY, 3, cv2.LINE_AA, tipLength=0.2)
        frac, too_far = 0.0, False
        if self.head_delta is not None:
            frac = head_progress(self.head_delta, direction, self.cfg.head_target_deg)
            axis = 0 if dx else 1
            too_far = abs(self.head_delta[axis]) > self.cfg.head_max_deg[axis]
        reached = frac >= self.cfg.head_reach_frac
        color = WARN if too_far else (COLLECT if reached else TARGET)
        k = float(np.clip(frac, 0.0, 1.0))
        if k > 0.02:
            end = (int(base[0] + (tip[0] - base[0]) * k), int(base[1] + (tip[1] - base[1]) * k))
            cv2.line(canvas, base, end, color, 10, cv2.LINE_AA)
        if too_far:
            self._centered(canvas, "Girou demais - volte um pouco", 180, WARN)


def _gating(calib_cfg: CalibrationConfig) -> dict:
    return dict(settle_max_s=calib_cfg.settle_max_s, fixation_window=calib_cfg.fixation_window,
                fixation_max_deg=calib_cfg.fixation_max_deg)


class ValidationProcedure(TargetProcedure):
    """Mede o erro em alvos NÃO usados na calibração.

    Dois tipos:

    * ``"imediata"`` (modo pesquisa): roda sozinha logo depois da
      calibração. Mede o erro do MODELO. É a única medida do desvio
      constante que aparece no uso: nas 15 sessões gravadas até 16/11, o erro
      em pontos novos da calibração (LOO) quase não previa o erro da
      validação (correlação 0,09), dominado por um desvio médio de ~106 px.
    * ``"reteste"`` (tecla T, mais tarde): mede o erro depois de um tempo de
      uso e corrige o desvio pela MÉDIA dos erros de todos os alvos (aplicada
      ao modelo, se não for grande demais).
    * ``"periodica"``: igual ao reteste, mas iniciada sozinha a cada
      ``ValidationConfig.revalidate_every_min`` minutos de uso (deriva ao
      longo da sessão, como a "drift check" do EyeLink).

    Métricas por olho (``por_olho``): a acurácia, a precisão e a
    disponibilidade (fração dos quadros com o olho aberto e o feixe
    atingindo a tela) de cada olho sozinho, além da binocular, como no
    relatório do Tobii Pro. Mostra se um olho é o problema (pálpebra,
    reflexo, iluminação de um lado).

    Além do erro bruto, reporta o erro "recentralizado": cada alvo corrigido
    pela média dos erros dos OUTROS alvos (validação cruzada). É o erro que
    sobra depois de uma recentralização com vários pontos. Reancorar por um
    ponto só (o centro, como até 16/11) piorava o erro em média (103 -> 134
    px nas validações gravadas); com 3 pontos ou mais, melhora (138 -> 96).

    O erro sai em px e em graus (com a distância do rosto medida pelo próprio
    pipeline, se ``--distance-cm`` não for dado) e por região da tela, com a
    precisão (RMS amostra a amostra e desvio-padrão) e a perda de dados.
    """

    title = "Validacao"

    def __init__(self, screen_size, model: CalibrationModel, cfg: ValidationConfig,
                 calib_cfg: CalibrationConfig, kind: str = "reteste"):
        targets = [(fx * screen_size[0], fy * screen_size[1]) for fx, fy in cfg.points]
        # Ordem aleatória: com a ordem fixa (centro sempre primeiro), o tempo
        # desde o procedimento anterior e a posição do alvo ficavam
        # confundidos (o primeiro alvo herdava o efeito da etapa da cabeça;
        # docs/metodologia.md, seção 5.11). Semente da calibração (+1000) para
        # a reprodução de sessões gravadas repetir a ordem.
        seed = None if calib_cfg.random_seed is None else calib_cfg.random_seed + 1000
        order = list(np.random.default_rng(seed).permutation(len(targets)))
        super().__init__(targets, screen_size, cfg.settle_s, cfg.collect_s,
                         calib_cfg.min_samples_per_point, order=order, **_gating(calib_cfg))
        self.cfg = cfg
        self.kind = kind
        self.title = {"imediata": "Validacao imediata",
                      "periodica": "Revalidacao periodica"}.get(kind, "Reteste")
        self.mad = calib_cfg.outlier_mad_threshold
        self.max_frac = calib_cfg.recenter_max_frac
        self.model = model
        self.results: list[PointResult] = []
        self.summary: dict = {}
        self.csv_path: Path | None = None
        self.applied_offset: np.ndarray | None = None

    def _width_cm(self) -> float:
        return float(self.cfg.screen_width_cm or self.model.screen.width_cm)

    def _px_to_deg(self, px: float, distance_cm: float | None) -> float | None:
        if not distance_cm or not np.isfinite(px):
            return None
        cm = px * self._width_cm() / self.screen_w
        return math.degrees(math.atan2(cm, distance_cm))

    def _region(self, target) -> str:
        fx, fy = target[0] / self.screen_w - 0.5, target[1] / self.screen_h - 0.5
        d = max(abs(fx), abs(fy))
        return "centro" if d < 0.1 else ("intermediaria" if d <= 0.25 else "borda")

    def _eye_metrics(self, per_eye: list, n_frames: int, deg) -> dict:
        """Acurácia, precisão e disponibilidade de cada olho sozinho.
        per_eye: por alvo, (n, 2 olhos, 2) previsões de TODAS as amostras
        válidas do alvo (NaN com o olho fechado); n_frames: quadros da coleta
        (válidos + perdidos)."""
        out = {}
        for k, name in enumerate(("direito", "esquerdo")):
            errs, precs, s2s, n_fin = [], [], [], 0
            for t, P in zip(self.targets, per_eye):
                p = P[:, k]
                p = p[np.all(np.isfinite(p), axis=1)]
                n_fin += len(p)
                if len(p) < 3:
                    continue
                errs.append(float(np.hypot(*(np.median(p, axis=0) - np.asarray(t)))))
                precs.append(float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1)))))
                s2s.append(float(np.sqrt(np.mean(np.sum(np.diff(p, axis=0) ** 2, 1)))))
            acc = float(np.mean(errs)) if errs else None
            prec = float(np.mean(precs)) if precs else None
            out[name] = {
                "acuracia_media_px": acc,
                "acuracia_media_graus": deg(acc) if acc is not None else None,
                "precisao_rms_media_px": prec,
                "precisao_rms_s2s_px": float(np.mean(s2s)) if s2s else None,
                "disponibilidade": n_fin / max(n_frames, 1),
                "n_alvos": len(errs),
            }
        return out

    def on_complete(self):
        rows, preds_all, n_ok = [], [], []
        precisions, s2s = [], []
        per_eye, n_frames = [], 0
        raw_data = CalibrationData()
        faces = []
        for i, target in enumerate(self.targets):
            feats, raws = self.arrays(i)
            raw_data.grid.append((target, feats, raws))
            raw_data.grid_obs.append(self.observations[i])
            raw_data.grid_times.append(np.asarray(self.times[i], dtype=float))
            n_frames += len(raws) + self.invalid[i]
            if len(raws):
                keep = outlier_mask(feats, self.mad)
                per_eye.append(self.model.predict_per_eye(raws)[keep])
                s, raws = feats[keep], raws[keep]
                faces.append(raws[:, RAW_FACE])
                preds = self.model.predict(raws)
                fin = np.all(np.isfinite(preds), axis=1)
                preds, s = preds[fin], s[fin]
            else:
                preds, s = np.zeros((0, 2)), np.zeros((0, 2))
                per_eye.append(np.zeros((0, 2, 2)))
            if not len(preds):
                preds = np.full((1, 2), np.nan)
            preds_all.append(np.median(preds, axis=0))
            n_ok.append(len(s))
            precisions.append(float(np.sqrt(np.mean(np.sum((preds - preds.mean(0)) ** 2, 1)))))
            if len(preds) >= 3:
                # RMS amostra a amostra (RMS-S2S): tremor quadro a quadro.
                s2s.append(float(np.sqrt(np.mean(np.sum(np.diff(preds, axis=0) ** 2, 1)))))
            for p, f in zip(preds, s):
                rows.append([i, target[0], target[1], p[0], p[1], f[0], f[1],
                             float(np.hypot(p[0] - target[0], p[1] - target[1]))])
        T = np.array(self.targets)
        P = np.array(preds_all)
        err = P - T
        dist_err = np.hypot(*err.T)
        self.results = [PointResult(tuple(t), (float(p[0]), float(p[1])), n)
                        for t, p, n in zip(self.targets, P, n_ok)]

        # Distância do rosto: a informada ou a medida pelo pipeline.
        distance = self.cfg.viewing_distance_cm
        if not distance and faces:
            distance = float(np.median(np.vstack(faces)[:, 2]))
        acc = float(np.nanmean(dist_err))
        # Desvio médio (constante) dos alvos e o erro sem ele (ótimo, dentro
        # da amostra) e recentralizado por validação cruzada (cada alvo
        # corrigido pela média dos OUTROS).
        ok = np.all(np.isfinite(err), axis=1)
        offset = np.nanmean(err, axis=0)
        acc_recentered = float(np.nanmean(np.hypot(*(err - offset).T)))
        loo = []
        for i in np.flatnonzero(ok):
            others = ok.copy()
            others[i] = False
            if others.sum() >= 2:
                loo.append(float(np.hypot(*(err[i] - err[others].mean(axis=0)))))
        acc_loo = float(np.mean(loo)) if loo else acc
        n_valid, n_invalid = int(sum(n_ok)), int(sum(self.invalid))
        data_loss = n_invalid / max(n_valid + n_invalid, 1)
        prec = float(np.nanmean(precisions))
        regions: dict = {}
        for t, e in zip(self.targets, dist_err):
            regions.setdefault(self._region(t), []).append(e)
        by_region = {k: float(np.nanmean(v)) for k, v in regions.items()}
        minutes = None
        if self.model.created_at:
            try:
                created = time.mktime(time.strptime(self.model.created_at, "%Y-%m-%d %H:%M:%S"))
                minutes = (time.time() - created) / 60.0
            except ValueError:
                pass
        deg = lambda px: self._px_to_deg(px, distance)  # noqa: E731
        self.summary = {
            "data": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tipo": self.kind,
            "minutos_desde_calibracao": minutes,
            "tela_px": [self.screen_w, self.screen_h],
            "distancia_cm": distance,
            "n_alvos": len(self.targets),
            "acuracia_media_px": acc,
            "acuracia_media_graus": deg(acc),
            "acuracia_mediana_px": float(np.nanmedian(dist_err)),
            "precisao_rms_media_px": prec,
            "precisao_rms_media_graus": deg(prec),
            "precisao_rms_s2s_px": float(np.mean(s2s)) if s2s else None,
            "perda_de_dados": data_loss,
            # Desvio médio (constante), a acurácia sem ele e a acurácia
            # recentralizada por validação cruzada.
            "deslocamento_medio_px": offset.tolist(),
            "deslocamento_medio_graus": deg(float(np.hypot(*offset))),
            "acuracia_sem_deslocamento_px": acc_recentered,
            "acuracia_recentralizada_px": acc_loo,
            "acuracia_recentralizada_graus": deg(acc_loo),
            "por_regiao_px": by_region,
            "por_olho": self._eye_metrics(per_eye, n_frames, deg),
            "calibracao": {k: v for k, v in self.model.info.items()
                           if k not in ("candidatos", "erro_loo_por_alvo_px", "erro_loo_alvos")},
            "por_alvo": [{"alvo": r.target, "estimativa": r.prediction, "regiao":
                          self._region(r.target), "erro_px": r.error_px, "n": r.n_samples,
                          "ordem": self.order.index(i) + 1}
                         for i, r in enumerate(self.results)],
        }

        # Reteste: corrige o desvio médio (recentralização com todos os
        # alvos), encolhido pela incerteza (shrink_offset).
        shrunk, factor = shrink_offset(err[ok]) if ok.sum() >= 2 else (offset, 1.0)
        shift = -shrunk
        if (self.kind in ("reteste", "periodica") and ok.sum() >= 3 and np.all(np.isfinite(shift))
                and abs(shift[0]) <= self.max_frac * self.screen_w
                and abs(shift[1]) <= self.max_frac * self.screen_h):
            self.model.offset = self.model.offset + shift
            self.applied_offset = shift
            self.summary["recentralizacao_aplicada_px"] = shift.tolist()
            self.summary["recentralizacao_fator"] = factor

        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.cfg.results_dir / f"validacao_{stamp}.csv"
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alvo_idx", "alvo_x", "alvo_y", "est_x", "est_y",
                        "yaw_deg", "pitch_deg", "erro_px"])
            w.writerows(rows)
        # Medidas brutas: permitem testar outros modelos nesta mesma sessão
        # (tools/reavaliar.py, tools/benchmark.py).
        raw_data.save(self.cfg.results_dir / f"validacao_{stamp}.npz")
        (self.cfg.results_dir / f"validacao_{stamp}.json").write_text(
            json.dumps(self.summary, indent=2, ensure_ascii=False), encoding="utf-8")

        self.succeeded = True
        fmt_deg = lambda px: (f" ({deg(px):.1f} graus)" if deg(px) is not None else "")  # noqa
        reg = "  ".join(f"{k} {v:.0f}" for k, v in sorted(by_region.items()))
        self.message_lines = [f"{self.title} concluida"
                              + (f" ({minutes:.0f} min apos calibrar)" if minutes else ""),
                              f"Erro medio: {acc:.0f} px{fmt_deg(acc)}   precisao RMS {prec:.0f} px"
                              f"   perda de dados {100 * data_loss:.0f}%",
                              f"Por regiao (px): {reg}",
                              f"Desvio constante: ({offset[0]:+.0f}, {offset[1]:+.0f}) px; "
                              f"recentralizado (validacao cruzada): {acc_loo:.0f} px"
                              f"{fmt_deg(acc_loo)}"]
        eyes = self.summary["por_olho"]
        self.message_lines.append("Por olho: " + "   ".join(
            f"{n} " + (f"{e['acuracia_media_px']:.0f} px" if e["acuracia_media_px"] is not None
                       else "-") + f" (disponivel {100 * e['disponibilidade']:.0f}%)"
            for n, e in eyes.items()))
        if self.kind in ("reteste", "periodica"):
            self.message_lines.append(
                "Desvio corrigido no cursor" if self.applied_offset is not None
                else "Desvio grande demais para corrigir: recalibre (ESPACO)")
        self.message_lines.append(f"Salvo em results/{self.csv_path.name}")
        print(f"[{self.title}] erro {acc:.1f} px{fmt_deg(acc)}, precisão RMS {prec:.1f} px, "
              f"desvio ({offset[0]:+.1f}, {offset[1]:+.1f}) px, recentralizado {acc_loo:.1f} px, "
              f"perda {100 * data_loss:.0f}%, por região {by_region}. "
              f"Resultados em {self.csv_path}")

    def render_result(self, canvas):
        _draw_point_results(canvas, self.results, tol_px=0.05 * self.screen_w)
        super().render_result(canvas)


class RecenterProcedure(TargetProcedure):
    """Correção rápida do desvio constante com poucos pontos (4), sem refazer
    a calibração.

    O pipeline geométrico já compensa rotação e translação da cabeça, mas
    pequenas mudanças de postura deixam um desvio constante que domina o
    erro no uso (validações gravadas: 138 px brutos, ~80 sem o desvio). Ele é
    corrigido somando um offset ao pixel do monitor real, estimado pela
    MÉDIA dos erros em vários alvos: com 1 ponto (o centro, até 16/11) a
    correção piorava o erro em média (138 -> 147 px); com 3 pontos, 96 px;
    com 5, 91 px.
    """

    title = "Recentralizar"
    result_timeout_s = 1.5

    def __init__(self, screen_size, model: CalibrationModel, cfg: CalibrationConfig,
                 kind: str = "manual"):
        targets = [(fx * screen_size[0], fy * screen_size[1]) for fx, fy in cfg.recenter_points]
        order = list(range(len(targets)))
        np.random.default_rng(cfg.random_seed).shuffle(order)
        super().__init__(targets, screen_size, cfg.settle_s, cfg.collect_s,
                         cfg.min_samples_per_point, order=order, **_gating(cfg))
        self.model = model
        self.mad = cfg.outlier_mad_threshold
        self.max_frac = cfg.recenter_max_frac
        # "manual" (R, barra), "inicio" (ao abrir com calibração salva) ou
        # "pos_calibracao" (conferência logo depois de calibrar).
        self.kind = kind
        self.applied: np.ndarray | None = None
        self.factor = 1.0
        if kind == "pos_calibracao":
            self.title = "Conferencia"

    def status_line(self) -> str:
        what = "Conferindo a calibracao" if self.kind == "pos_calibracao" else "Recentralizar"
        return f"{what}: olhe para o ponto ({self.pos + 1}/{len(self.order)})  -  ESC cancela"

    def on_complete(self):
        deltas, raws_all = [], []
        for i, target in enumerate(self.targets):
            feats, raws = self.arrays(i)
            if len(raws) < 3:
                continue
            raws = raws[outlier_mask(feats, self.mad)]
            pred = np.nanmedian(self.model.predict(raws), axis=0)
            if np.all(np.isfinite(pred)):
                deltas.append(np.array(target) - pred)
                raws_all.append(raws)
        mean = np.mean(deltas, axis=0) if len(deltas) >= 2 else np.full(2, np.nan)
        delta = mean
        if len(deltas) >= 2:
            # Desvio encolhido pela incerteza da média (shrink_offset).
            delta, self.factor = shrink_offset(np.asarray(deltas))
        raws = np.vstack(raws_all) if raws_all else np.zeros((0, RAW_SIZE))
        if (not np.all(np.isfinite(mean)) or abs(mean[0]) > self.max_frac * self.screen_w
                or abs(mean[1]) > self.max_frac * self.screen_h):
            # Desvio grande demais para ser corrigido por um deslocamento:
            # a postura mudou muito ou o rosto foi mal detectado. Aplicar
            # mesmo assim deslocava o cursor inteiro (caso real: -964 px no
            # eixo y, 23 s depois de calibrar) e deixava parte da tela
            # inalcançável quando a pessoa voltava à postura normal.
            self.succeeded = False
            self.result_timeout_s = 6.0   # tempo para ler o aviso
            self.message_lines = [
                f"Desvio muito grande ({mean[0]:+.0f}, {mean[1]:+.0f}) px: NAO aplicado",
                "A postura mudou demais ou o rosto nao foi bem detectado.",
                "Volte a postura da calibracao ou recalibre (ESPACO)."]
            print(f"[Recentralizar] Desvio de ({mean[0]:+.1f}, {mean[1]:+.1f}) px rejeitado "
                  f"(limite {self.max_frac:.0%} da tela).")
            return
        self.model.offset = self.model.offset + delta
        self.applied = delta
        self.model.info[f"recentralizacao_{self.kind}_px"] = [float(v) for v in delta]
        # O offset agora vale para a postura atual: ela vira a referência do
        # aviso de postura (calibration.CalibrationModel.posture_deviation).
        self.model.head_ref = np.median(head_angles(raws), axis=0)
        self.model.pos_ref = np.median(raws[:, RAW_FACE], axis=0)
        self.succeeded = True
        spread = float(np.mean([np.hypot(*(d - mean)) for d in deltas]))
        self.message_lines = [f"Recentralizado: deslocamento ({delta[0]:+.0f}, {delta[1]:+.0f}) px"
                              f" (dispersao entre os pontos {spread:.0f} px)"]
        print(f"[Recentralizar] Offset ajustado em ({delta[0]:+.1f}, {delta[1]:+.1f}) px "
              f"(média ({mean[0]:+.1f}, {mean[1]:+.1f}), fator {self.factor:.2f}).")
