import numpy as np
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve


class BestThreshold:
    @staticmethod
    def _best_threshold(y_true, y_score, technique="max_f1", max_candidates=1024):
        """
        Zwraca: (best_threshold, best_value)

        technique:
          - "max_f1"  : próg maksymalizujący F1
          - "youden"  : próg maksymalizujący J = TPR - FPR (aliasy: younden, younden)
          - "dist01"  : próg minimalizujący dystans do (0,1) na ROC
        """
        y_true = np.asarray(y_true).astype(np.int32).ravel()
        y_score = np.asarray(y_score).astype(np.float64).ravel()

        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError("y_true i y_score muszą mieć ten sam rozmiar.")
        if y_true.size == 0:
            raise ValueError("Puste wejście.")
        if not np.all(np.isin(np.unique(y_true), [0, 1])):
            raise ValueError("y_true musi być binarne (0/1).")
        if np.unique(y_true).size < 2:
            # Nie da się policzyć ROC/F1 sensownie bez obu klas
            return 0.5, float("nan")

        tech = technique.lower().strip()

        # ===== 1) MAX F1 =====
        if tech in {"max_f1", "f1"}:
            # Dokładniej niż skan linspace: progi z PR curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)

            if thresholds.size == 0:
                return 0.5, float("nan")

            # precision/recall mają o 1 element więcej niż thresholds
            f1_vals = 2.0 * precision * recall / (precision + recall + 1e-12)
            f1_vals = f1_vals[:-1]

            # opcjonalne ograniczenie liczby kandydatów (dla bardzo dużych danych)
            if max_candidates is not None and thresholds.size > max_candidates:
                idxs = np.linspace(0, thresholds.size - 1, int(max_candidates), dtype=int)
                thresholds_sub = thresholds[idxs]
                f1_sub = f1_vals[idxs]
                best_idx_sub = int(np.nanargmax(f1_sub))
                return float(thresholds_sub[best_idx_sub]), float(f1_sub[best_idx_sub])

            best_idx = int(np.nanargmax(f1_vals))
            return float(thresholds[best_idx]), float(f1_vals[best_idx])

        # ===== 2) YOUDEN =====
        if tech in {"youden", "younden", "younden", "j"}:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)

            # roc_curve potrafi zwrócić inf jako pierwszy threshold
            finite = np.isfinite(thresholds)
            fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]

            if thresholds.size == 0:
                return 0.5, float("nan")

            j_stat = tpr - fpr
            best_idx = int(np.nanargmax(j_stat))
            return float(thresholds[best_idx]), float(j_stat[best_idx])

        # ===== 3) DIST01 =====
        if tech in {"dist01", "dist_01", "d01"}:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)

            finite = np.isfinite(thresholds)
            fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]

            if thresholds.size == 0:
                return 0.5, float("nan")

            # dystans do punktu idealnego (0,1)
            dist = np.sqrt((1.0 - tpr) ** 2 + (fpr ** 2))
            best_idx = int(np.nanargmin(dist))
            return float(thresholds[best_idx]), float(dist[best_idx])

        raise ValueError("Nieznana technika. Użyj: 'max_f1', 'youden', 'dist01'.")


    @staticmethod
    def all_thresholds(y_true, y_score, max_candidates=1024):
        """
        Pomocniczo: policz wszystkie 3 metody naraz.
        """
        t_f1, v_f1 = BestThreshold._best_threshold(
            y_true, y_score, technique="max_f1", max_candidates=max_candidates
        )
        t_y, v_y = BestThreshold._best_threshold(
            y_true, y_score, technique="youden", max_candidates=max_candidates
        )
        t_d, v_d = BestThreshold._best_threshold(
            y_true, y_score, technique="dist01", max_candidates=max_candidates
        )
        return {
            "max_f1": {"threshold": t_f1, "value": v_f1},
            "youden": {"threshold": t_y, "value": v_y},
            "dist01": {"threshold": t_d, "value": v_d},
        }
