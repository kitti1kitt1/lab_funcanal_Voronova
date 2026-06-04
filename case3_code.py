#Кейс 3: изометрия и матрица расстояний.

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "case3_report_assets"
OUT.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)




def pairwise_distances(Y: np.ndarray) -> np.ndarray:
    """Вычисляет матрицу попарных евклидовых расстояний для строк матрицы Y."""
    diff = Y[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def check_distance_matrix(D: np.ndarray, tol: float = 1e-10, sample_triples: int | None = None, seed: int = 0) -> dict:
    n = D.shape[0]
    diag_error = float(np.max(np.abs(np.diag(D))))
    symmetry_error = float(np.linalg.norm(D - D.T, ord="fro"))
    min_value = float(np.min(D))
    nonnegative = bool(min_value >= -tol)

    if sample_triples is None:
        total = n ** 3
        violations = 0
        max_violation = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    value = D[i, j] - D[i, k] - D[k, j]
                    if value > tol:
                        violations += 1
                        max_violation = max(max_violation, float(value))
    else:
        local_rng = np.random.default_rng(seed)
        total = sample_triples
        violations = 0
        max_violation = 0.0
        for _ in range(sample_triples):
            i, j, k = local_rng.integers(0, n, size=3)
            value = D[i, j] - D[i, k] - D[k, j]
            if value > tol:
                violations += 1
                max_violation = max(max_violation, float(value))

    return {
        "diag max abs": diag_error,
        "symmetry Frobenius error": symmetry_error,
        "min entry": min_value,
        "nonnegative": nonnegative,
        "triangle violations": violations,
        "checked triples": total,
        "violation share": violations / total,
        "max triangle violation": max_violation,
    }


def double_centering(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    return D2, J, B


def spectral_decomposition(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def gram_spectrum_summary(eigvals: np.ndarray, tol: float = 1e-9) -> dict:
    positive = int(np.sum(eigvals > tol))
    zero = int(np.sum(np.abs(eigvals) <= tol))
    negative = int(np.sum(eigvals < -tol))
    return {
        "positive eigenvalues": positive,
        "zero eigenvalues": zero,
        "negative eigenvalues": negative,
        "rank estimate": positive,
        "min eigenvalue": float(np.min(eigvals)),
        "max eigenvalue": float(np.max(eigvals)),
    }


def recover_coordinates(eigvals: np.ndarray, eigvecs: np.ndarray, m: int | None = None, tol: float = 1e-9) -> np.ndarray:
    pos = eigvals > tol
    if m is None:
        selected = np.where(pos)[0]
    else:
        selected = np.where(pos)[0][:m]
    if len(selected) == 0:
        return np.zeros((len(eigvals), 0))
    lambdas = eigvals[selected]
    U = eigvecs[:, selected]
    return U * np.sqrt(lambdas)[None, :]


def center_points(Y: np.ndarray) -> np.ndarray:
    return Y - Y.mean(axis=0, keepdims=True)


def procrustes_align(Y_est: np.ndarray, Y_true: np.ndarray) -> tuple[np.ndarray, float]:
    X = center_points(Y_est)
    Y = center_points(Y_true)
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    X_aligned = X @ R
    rmse = float(np.sqrt(np.mean((X_aligned - Y) ** 2)))
    return X_aligned, rmse


def distance_errors(D: np.ndarray, Y: np.ndarray) -> dict:
    D_hat = pairwise_distances(Y)
    diff = D - D_hat
    E_max = float(np.max(np.abs(diff)))
    E_F = float(np.linalg.norm(diff, ord="fro"))
    E_rel = float(E_F / np.linalg.norm(D, ord="fro"))
    corr = float(np.corrcoef(D[np.triu_indices_from(D, k=1)], D_hat[np.triu_indices_from(D_hat, k=1)])[0, 1])
    return {
        "D_hat": D_hat,
        "E_max": E_max,
        "E_F": E_F,
        "E_rel": E_rel,
        "corr": corr,
    }


def add_symmetric_noise_to_distances(D: np.ndarray, epsilon: float, seed: int = 0) -> np.ndarray:
    local_rng = np.random.default_rng(seed)
    n = D.shape[0]
    R = local_rng.normal(size=(n, n))
    R = (R + R.T) / 2
    np.fill_diagonal(R, 0.0)
    D_eps = D + epsilon * R
    D_eps = (D_eps + D_eps.T) / 2
    np.fill_diagonal(D_eps, 0.0)
    D_eps[D_eps < 0] = 0.0
    np.fill_diagonal(D_eps, 0.0)
    return D_eps


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()




n = 60
true_dim = 3
Y_true = rng.normal(loc=0.0, scale=1.0, size=(n, true_dim))
Y_true_centered = center_points(Y_true)
D = pairwise_distances(Y_true)




base_checks = check_distance_matrix(D, tol=1e-10, sample_triples=None)
pd.DataFrame([base_checks]).to_csv(OUT / "table_distance_checks.csv", index=False)




D2, J, B = double_centering(D)
B_symmetry_error = np.linalg.norm(B - B.T, ord="fro")
eigvals, eigvecs = spectral_decomposition(B)
spectrum_info = gram_spectrum_summary(eigvals, tol=1e-9)
spectrum_info["B symmetry Frobenius error"] = float(B_symmetry_error)
pd.DataFrame([spectrum_info]).to_csv(OUT / "table_gram_spectrum_summary.csv", index=False)

pd.DataFrame({
    "index": np.arange(1, min(12, len(eigvals)) + 1),
    "eigenvalue": eigvals[:min(12, len(eigvals))]
}).to_csv(OUT / "table_first_eigenvalues.csv", index=False)




Y_full = recover_coordinates(eigvals, eigvecs, m=None, tol=1e-9)
Y_2 = recover_coordinates(eigvals, eigvecs, m=2, tol=1e-9)
Y_3 = recover_coordinates(eigvals, eigvecs, m=3, tol=1e-9)

Y_3_aligned, coord_rmse = procrustes_align(Y_3, Y_true_centered)

iso_rows = []
for label, Y in [("full positive rank", Y_full), ("2D", Y_2), ("3D", Y_3)]:
    errs = distance_errors(D, Y)
    iso_rows.append({
        "embedding": label,
        "dimension": Y.shape[1],
        "E_max": errs["E_max"],
        "E_F": errs["E_F"],
        "E_rel": errs["E_rel"],
        "distance correlation": errs["corr"],
    })
iso_df = pd.DataFrame(iso_rows)
iso_df.to_csv(OUT / "table_isometry_errors.csv", index=False)

pd.DataFrame([{"Procrustes RMSE for 3D coordinates": coord_rmse}]).to_csv(OUT / "table_procrustes.csv", index=False)




mmax = min(10, spectrum_info["positive eigenvalues"])
dim_rows = []
for m in range(1, mmax + 1):
    Y_m = recover_coordinates(eigvals, eigvecs, m=m, tol=1e-9)
    errs = distance_errors(D, Y_m)
    captured = float(np.sum(eigvals[:m]) / np.sum(eigvals[eigvals > 1e-9]))
    dim_rows.append({
        "m": m,
        "E_F": errs["E_F"],
        "E_rel": errs["E_rel"],
        "distance correlation": errs["corr"],
        "share of positive spectrum": captured,
    })
dim_df = pd.DataFrame(dim_rows)
dim_df.to_csv(OUT / "table_dimension_errors.csv", index=False)




epsilon_values = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
noise_rows = []
base_Y_ref = Y_full
for eps in epsilon_values:
    D_eps = add_symmetric_noise_to_distances(D, eps, seed=RNG_SEED + 100)
    checks_eps = check_distance_matrix(D_eps, tol=1e-10, sample_triples=20000, seed=RNG_SEED)
    _, _, B_eps = double_centering(D_eps)
    evals_eps, evecs_eps = spectral_decomposition(B_eps)
    sp_eps = gram_spectrum_summary(evals_eps, tol=1e-8)
    Y_eps = recover_coordinates(evals_eps, evecs_eps, m=3, tol=1e-8)
    errs_eps = distance_errors(D_eps, Y_eps)
    D_hat_eps = errs_eps["D_hat"]
    rel_to_true = float(np.linalg.norm(D - D_hat_eps, ord="fro") / np.linalg.norm(D, ord="fro"))

    noise_rows.append({
        "epsilon": eps,
        "negative eigenvalues": sp_eps["negative eigenvalues"],
        "min eigenvalue": sp_eps["min eigenvalue"],
        "positive eigenvalues": sp_eps["positive eigenvalues"],
        "triangle violation share": checks_eps["violation share"],
        "E_rel to noisy D": errs_eps["E_rel"],
        "E_rel to true D": rel_to_true,
        "distance correlation": errs_eps["corr"],
    })
noise_df = pd.DataFrame(noise_rows)
noise_df.to_csv(OUT / "table_noise_sensitivity.csv", index=False)




plt.figure(figsize=(7, 4.5))
plt.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", markersize=3, linewidth=1)
plt.axhline(0, linewidth=1)
plt.xlabel("Номер собственного значения")
plt.ylabel("Собственное значение")
plt.title("Спектр матрицы Грама B")
plt.grid(True)
save_fig(OUT / "fig_gram_spectrum.png")

plt.figure(figsize=(6, 5))
plt.scatter(Y_2[:, 0], Y_2[:, 1], s=28)
for i in range(min(10, n)):
    plt.text(Y_2[i, 0], Y_2[i, 1], str(i), fontsize=8)
plt.xlabel("компонента 1")
plt.ylabel("компонента 2")
plt.title("Двумерное MDS-вложение")
plt.grid(True)
save_fig(OUT / "fig_embedding_2d.png")

plt.figure(figsize=(6, 5))
plt.scatter(Y_true_centered[:, 0], Y_true_centered[:, 1], s=28, label="исходные")
plt.scatter(Y_3_aligned[:, 0], Y_3_aligned[:, 1], s=20, marker="x", label="восстановленные")
plt.xlabel("координата 1")
plt.ylabel("координата 2")
plt.title("Исходные и восстановленные координаты после Procrustes")
plt.legend()
plt.grid(True)
save_fig(OUT / "fig_procrustes_2d_projection.png")

plt.figure(figsize=(7, 4.5))
plt.plot(dim_df["m"], dim_df["E_F"], marker="o")
plt.xlabel("Размерность вложения m")
plt.ylabel("Ошибка E_F")
plt.title("Ошибка восстановления расстояний в зависимости от размерности")
plt.grid(True)
save_fig(OUT / "fig_dimension_error.png")

positive_eigs = eigvals[eigvals > 1e-9]
plt.figure(figsize=(7, 4.5))
plt.plot(np.arange(1, len(positive_eigs) + 1), positive_eigs, marker="o")
plt.xlabel("Номер положительного собственного значения")
plt.ylabel("Значение")
plt.yscale("log")
plt.title("Положительный спектр B в логарифмической шкале")
plt.grid(True)
save_fig(OUT / "fig_positive_spectrum_log.png")

for label, Y, filename in [("2D", Y_2, "fig_distance_scatter_2d.png"), ("3D", Y_3, "fig_distance_scatter_3d.png")]:
    D_hat = pairwise_distances(Y)
    iu = np.triu_indices_from(D, k=1)
    plt.figure(figsize=(5.5, 5))
    plt.scatter(D[iu], D_hat[iu], s=12, alpha=0.7)
    max_val = max(float(np.max(D[iu])), float(np.max(D_hat[iu])))
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Исходные расстояния")
    plt.ylabel("Восстановленные расстояния")
    plt.title(f"Исходные и восстановленные расстояния: {label}")
    plt.grid(True)
    save_fig(OUT / filename)

plt.figure(figsize=(7, 4.5))
plt.plot(noise_df["epsilon"], noise_df["min eigenvalue"], marker="o")
plt.axhline(0, linewidth=1)
plt.xlabel("epsilon")
plt.ylabel("Минимальное собственное значение")
plt.title("Влияние шума на минимальное собственное значение B")
plt.grid(True)
save_fig(OUT / "fig_noise_min_eigenvalue.png")

plt.figure(figsize=(7, 4.5))
plt.plot(noise_df["epsilon"], noise_df["negative eigenvalues"], marker="o")
plt.xlabel("epsilon")
plt.ylabel("Число отрицательных собственных значений")
plt.title("Влияние шума на число отрицательных собственных значений")
plt.grid(True)
save_fig(OUT / "fig_noise_negative_count.png")

plt.figure(figsize=(7, 4.5))
plt.plot(noise_df["epsilon"], noise_df["E_rel to noisy D"], marker="o", label="к шумной D")
plt.plot(noise_df["epsilon"], noise_df["E_rel to true D"], marker="s", label="к исходной D")
plt.xlabel("epsilon")
plt.ylabel("Относительная ошибка")
plt.title("Влияние шума на ошибку восстановления")
plt.legend()
plt.grid(True)
save_fig(OUT / "fig_noise_reconstruction_error.png")

print("Результаты сохранены в:", OUT)
print("Проверка D:")
print(pd.DataFrame([base_checks]).T)
print("\nСпектр B:")
print(pd.DataFrame([spectrum_info]).T)
print("\nОшибки изометричности:")
print(iso_df)
print("\nУстойчивость к шуму:")
print(noise_df)
