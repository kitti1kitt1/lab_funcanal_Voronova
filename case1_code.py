#Кейс 1. Интегральный оператор с ядром K(x,t)=min(x,t).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "case1_report_assets"
OUT.mkdir(parents=True, exist_ok=True)




def lambda_exact(k):
    k = np.asarray(k)
    return 1.0 / (np.pi**2 * (k - 0.5)**2)

def phi_exact(k, x):
    return np.sqrt(2) * np.sin((k - 0.5) * np.pi * x)

def uniform_grid(n):
    return np.linspace(0, 1, n)

def nonuniform_grid(n):
    s = np.linspace(0, 1, n)
    return s**2

def kernel_matrix(x):
    return np.minimum(x[:, None], x[None, :])

def rectangle_weights(n):
    return np.full(n, 1.0/n)

def trapezoid_weights(n):
    h = 1.0/(n-1)
    w = np.full(n, h)
    w[0] = h/2
    w[-1] = h/2
    return w

def simpson_weights(n):
    if (n-1) % 2 != 0:
        raise ValueError('For Simpson rule n must be odd')
    h = 1.0/(n-1)
    w = np.zeros(n)
    w[0] = h/3
    w[-1] = h/3
    for j in range(1, n-1):
        w[j] = 4*h/3 if j % 2 else 2*h/3
    return w

def nonuniform_trapezoid_weights(x):
    n = len(x)
    w = np.zeros(n)
    w[0] = (x[1] - x[0])/2
    w[-1] = (x[-1] - x[-2])/2
    for j in range(1, n-1):
        w[j] = (x[j+1] - x[j-1])/2
    return w

def build_matrices(x, w, K=None):
    if K is None:
        K = kernel_matrix(x)
    A = K * w[None, :]
    sw = np.sqrt(w)
    B = sw[:, None] * K * sw[None, :]
    return K, A, B

def spectrum_B(B):
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]

def recover_phi(eigenvectors, w):
    return eigenvectors / np.sqrt(w)[:, None]

def l2_norm(phi, w):
    return np.sqrt(np.sum(phi**2 * w))

def normalize_columns(Phi, w):
    Phi = Phi.copy()
    for j in range(Phi.shape[1]):
        nrm = l2_norm(Phi[:, j], w)
        if nrm > 0:
            Phi[:, j] /= nrm
    return Phi

def exact_phi_matrix(x, kmax):
    P = np.zeros((len(x), kmax))
    for k in range(1, kmax+1):
        P[:, k-1] = phi_exact(k, x)
    return P

def orient_by_inner_product(Phi_num, Phi_ref, w):
    Phi = Phi_num.copy()
    for k in range(Phi_ref.shape[1]):
        if np.sum(Phi[:, k] * Phi_ref[:, k] * w) < 0:
            Phi[:, k] *= -1
    return Phi

def spectrum_solution(n, weights='trapezoid', x=None, K=None, kmax=5):
    if x is None:
        x = uniform_grid(n)
    if weights == 'trapezoid':
        w = trapezoid_weights(len(x))
    elif weights == 'rectangle':
        w = rectangle_weights(len(x))
    elif weights == 'simpson':
        w = simpson_weights(len(x))
    elif weights == 'nonuniform_trapezoid':
        w = nonuniform_trapezoid_weights(x)
    else:
        raise ValueError(weights)
    K, A, B = build_matrices(x, w, K=K)
    vals, vecs = spectrum_B(B)
    Phi = normalize_columns(recover_phi(vecs, w), w)
    Phi_exact = exact_phi_matrix(x, kmax)
    Phi_oriented = orient_by_inner_product(Phi[:, :kmax], Phi_exact, w)
    return x, w, K, A, B, vals, Phi_oriented, Phi_exact




nG = 51
xG = uniform_grid(nG)
G = kernel_matrix(xG)
eigsG = np.linalg.eigvalsh(G)
summary_gram = pd.DataFrame({
    'Показатель': ['||G - G^T||_F', 'lambda_min(G)', 'lambda_max(G)', 'все lambda >= -1e-12'],
    'Значение': [np.linalg.norm(G-G.T), eigsG.min(), eigsG.max(), str(bool(np.all(eigsG >= -1e-12)))]
})
summary_gram.to_csv(OUT/'table_gram_summary.csv', index=False)




plt.figure(figsize=(5,4))
plt.imshow(G, origin='lower', extent=[0,1,0,1], aspect='auto')
plt.colorbar(label='G_ij')
plt.xlabel('x_j')
plt.ylabel('x_i')
plt.title('Матрица Грама G_ij = min(x_i, x_j)')
plt.tight_layout()
plt.savefig(OUT/'fig_gram_heatmap.png', dpi=200)
plt.close()




vals_desc_G = eigsG[::-1]
plt.figure(figsize=(5.5,4))
plt.plot(np.arange(1,len(vals_desc_G)+1), vals_desc_G, marker='o', markersize=3)
plt.xlabel('Номер собственного значения')
plt.ylabel('Собственное значение')
plt.title('Спектр матрицы Грама')
plt.grid(True, alpha=.35)
plt.tight_layout()
plt.savefig(OUT/'fig_gram_spectrum.png', dpi=200)
plt.close()




x101, w101, K101, A101, B101, vals101, Phi101, PhiE101 = spectrum_solution(101, kmax=5)
rows = []
for k in range(1,6):
    rows.append({'k': k, 'lambda_exact': lambda_exact(k).item(), 'lambda_numeric_n101': vals101[k-1], 'abs_error': abs(vals101[k-1]-lambda_exact(k).item())})
pd.DataFrame(rows).to_csv(OUT/'table_eigenvalues_n101.csv', index=False)




x200, w200, K200, A200, B200, vals200, Phi200, PhiE200 = spectrum_solution(200, kmax=5)
for k in range(1,4):
    plt.figure(figsize=(5.5,4))
    plt.plot(x200, PhiE200[:,k-1], label=f'analytic phi_{k}')
    plt.plot(x200, Phi200[:,k-1], linestyle='--', label=f'numeric phi_{k}')
    plt.xlabel('x')
    plt.ylabel(f'phi_{k}(x)')
    plt.title(f'Сравнение собственной функции phi_{k}, n=200')
    plt.grid(True, alpha=.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT/f'fig_phi{k}_comparison.png', dpi=200)
    plt.close()




n_values = [20,50,100,200]
kmax = 5
all_rows = []
solutions = {}
for n in n_values:
    x,w,K,A,B,vals,Phi,PhiE = spectrum_solution(n, kmax=kmax)
    solutions[n] = (x,w,K,A,B,vals,Phi,PhiE)
    for k in range(1,kmax+1):
        lam_ex = lambda_exact(k).item()
        lam_num = vals[k-1]
        phi_err = l2_norm(Phi[:,k-1] - PhiE[:,k-1], w)
        all_rows.append({'n': n, 'k': k, 'lambda_exact': lam_ex, 'lambda_numeric': lam_num, 'lambda_error': abs(lam_num-lam_ex), 'phi_L2_error': phi_err})
df_errors = pd.DataFrame(all_rows)
df_errors.to_csv(OUT/'table_errors_long.csv', index=False)
df_errors.pivot(index='n', columns='k', values='lambda_error').to_csv(OUT/'table_lambda_error_pivot.csv')
df_errors.pivot(index='n', columns='k', values='phi_L2_error').to_csv(OUT/'table_phi_error_pivot.csv')




plt.figure(figsize=(5.5,4))
for k in range(1,kmax+1):
    part = df_errors[df_errors['k']==k]
    plt.plot(part['n'], part['lambda_error'], marker='o', label=f'k={k}')
plt.xlabel('Размер сетки n')
plt.ylabel('e_lambda')
plt.yscale('log')
plt.title('Ошибка собственных значений')
plt.grid(True, alpha=.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'fig_lambda_error_by_n.png', dpi=200)
plt.close()

plt.figure(figsize=(5.5,4))
for k in range(1,kmax+1):
    part = df_errors[df_errors['k']==k]
    plt.plot(part['n'], part['phi_L2_error'], marker='o', label=f'k={k}')
plt.xlabel('Размер сетки n')
plt.ylabel('e_phi')
plt.yscale('log')
plt.title('Ошибка собственных функций')
plt.grid(True, alpha=.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'fig_phi_error_by_n.png', dpi=200)
plt.close()




def K_truncated(x, m):
    Km = np.zeros((len(x),len(x)))
    for k in range(1,m+1):
        phi = phi_exact(k, x)
        Km += lambda_exact(k).item() * np.outer(phi, phi)
    return Km
m_values = [1,2,5,10]
K_exact = K200
fro_rows=[]
for m in m_values:
    Km = K_truncated(x200, m)
    fro = np.linalg.norm(K_exact-Km, 'fro')
    rel = fro/np.linalg.norm(K_exact,'fro')
    fro_rows.append({'m':m, 'frobenius_error':fro, 'relative_error':rel})
    plt.figure(figsize=(5,4))
    plt.imshow(Km, origin='lower', extent=[0,1,0,1], aspect='auto')
    plt.colorbar(label=f'K_{m}(x,t)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'Усечённое приближение K_{m}')
    plt.tight_layout()
    plt.savefig(OUT/f'fig_Km_{m}.png', dpi=200)
    plt.close()

pd.DataFrame(fro_rows).to_csv(OUT/'table_frobenius.csv', index=False)

plt.figure(figsize=(5,4))
plt.imshow(K_exact, origin='lower', extent=[0,1,0,1], aspect='auto')
plt.colorbar(label='K(x,t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Точное ядро K(x,t)=min(x,t)')
plt.tight_layout()
plt.savefig(OUT/'fig_kernel_exact.png', dpi=200)
plt.close()

plt.figure(figsize=(5.5,4))
plt.plot([r['m'] for r in fro_rows], [r['relative_error'] for r in fro_rows], marker='o')
plt.xlabel('m')
plt.ylabel('Относительная ошибка Фробениуса')
plt.title('Ошибка усечённого разложения')
plt.grid(True, alpha=.35)
plt.tight_layout()
plt.savefig(OUT/'fig_frobenius_error.png', dpi=200)
plt.close()




quad_methods = {'Прямоугольники': 'rectangle', 'Трапеции': 'trapezoid', 'Симпсон': 'simpson'}
quad_rows=[]
for name, meth in quad_methods.items():
    x,w,K,A,B,vals,Phi,PhiE = spectrum_solution(101, weights=meth, kmax=kmax)
    for k in range(1,kmax+1):
        quad_rows.append({'method': name, 'k': k, 'lambda_error': abs(vals[k-1]-lambda_exact(k).item()), 'phi_L2_error': l2_norm(Phi[:,k-1]-PhiE[:,k-1], w)})
df_quad = pd.DataFrame(quad_rows)
df_quad.to_csv(OUT/'table_quadrature.csv', index=False)

plt.figure(figsize=(5.5,4))
for name in quad_methods:
    part = df_quad[df_quad['method']==name]
    plt.plot(part['k'], part['lambda_error'], marker='o', label=name)
plt.xlabel('k')
plt.ylabel('Ошибка lambda')
plt.yscale('log')
plt.title('Влияние квадратурной формулы')
plt.grid(True, alpha=.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'fig_quadrature_lambda_error.png', dpi=200)
plt.close()




x_uni = uniform_grid(101); w_uni = trapezoid_weights(101)
x_non = nonuniform_grid(101); w_non = nonuniform_trapezoid_weights(x_non)
grid_rows=[]
for name, x, w, weight_label in [('Равномерная', x_uni, w_uni, 'custom'), ('Неравномерная', x_non, w_non, 'custom')]:
    K,A,B = build_matrices(x, w)
    vals, vecs = spectrum_B(B)
    Phi = normalize_columns(recover_phi(vecs, w), w)
    PhiE = exact_phi_matrix(x,kmax)
    Phi = orient_by_inner_product(Phi[:,:kmax], PhiE, w)
    for k in range(1,kmax+1):
        grid_rows.append({'grid': name, 'k': k, 'lambda_error': abs(vals[k-1]-lambda_exact(k).item()), 'phi_L2_error': l2_norm(Phi[:,k-1]-PhiE[:,k-1],w)})
df_grid = pd.DataFrame(grid_rows)
df_grid.to_csv(OUT/'table_grid_compare.csv', index=False)

plt.figure(figsize=(6,2.7))
plt.plot(x_uni, np.zeros_like(x_uni), marker='o', linestyle='', markersize=3, label='Равномерная')
plt.plot(x_non, np.ones_like(x_non), marker='o', linestyle='', markersize=3, label='Неравномерная')
plt.yticks([0,1], ['равномерная','неравномерная'])
plt.xlabel('x')
plt.title('Расположение узлов сетки')
plt.grid(True, alpha=.35)
plt.tight_layout()
plt.savefig(OUT/'fig_grid_nodes.png', dpi=200)
plt.close()

rng = np.random.default_rng(42)
R = rng.normal(size=(101,101)); R = (R+R.T)/2; R = R/np.max(np.abs(R))
xp = uniform_grid(101); wp=trapezoid_weights(101); Kbase=kernel_matrix(xp)
K,A,B=build_matrices(xp, wp, Kbase); vals0, vecs0 = spectrum_B(B); Phi0=normalize_columns(recover_phi(vecs0,wp),wp); PhiE=exact_phi_matrix(xp,kmax); Phi0=orient_by_inner_product(Phi0[:,:kmax],PhiE,wp)
eps_values=[0,0.001,0.01,0.05]
pert_rows=[]
for eps in eps_values:
    Keps=Kbase+eps*R
    K,A,B=build_matrices(xp,wp,Keps); vals,vecs=spectrum_B(B); Phi=normalize_columns(recover_phi(vecs,wp),wp); Phi=orient_by_inner_product(Phi[:,:kmax],Phi0,wp)
    for k in range(1,kmax+1):
        pert_rows.append({'epsilon':eps,'k':k,'lambda_change':abs(vals[k-1]-vals0[k-1]),'phi_change':l2_norm(Phi[:,k-1]-Phi0[:,k-1],wp)})
df_pert=pd.DataFrame(pert_rows)
df_pert.to_csv(OUT/'table_perturbation.csv', index=False)

plt.figure(figsize=(5.5,4))
for k in range(1,kmax+1):
    part = df_pert[df_pert['k']==k]
    plt.plot(part['epsilon'], part['lambda_change'], marker='o', label=f'k={k}')
plt.xlabel('epsilon')
plt.ylabel('Изменение lambda')
plt.yscale('log')
plt.title('Влияние возмущения ядра')
plt.grid(True, alpha=.35)
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'fig_perturb_lambda_change.png', dpi=200)
plt.close()

print('Assets generated in', OUT)
