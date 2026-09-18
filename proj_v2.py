"""
Higher-Order Correlation Power Analysis (HOCPA) on Masked Kyber
with Bounded Adaptive Central Reduction Randomization
-- v2: vectorized attack, more trials, additional performance metrics --
"""

import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

Q = 3329
KEY_SPACE = 256
NUM_TRACES = 5000
NUM_TRACES_RMAX = 20000

NOISE_STD = 0.18
BETA = 0.18

LAMBDA_FIXED = 0.5

LAMBDA_INIT = 0.3
LAMBDA_MAX = 0.70
LAMBDA_MIN = 0.20
ADAPTIVE_THRESHOLD = 0.048
ADAPTIVE_STEP_UP = 0.012
ADAPTIVE_STEP_DOWN = 0.018

NUM_TRIALS = 200
TRACE_STEP = 500


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hamming_weight(x):
    x = np.atleast_1d(x).astype(np.int64)
    flat = x.reshape(-1)
    hw_flat = np.array([bin(int(v) & 0xFFFF).count('1') for v in flat])
    return hw_flat.reshape(x.shape)


def central_reduction(x, q=Q):
    x = np.atleast_1d(x).astype(np.int64)
    x_mod = x % q
    return np.where(x_mod > q // 2, x_mod - q, x_mod)


def noncentral_reduction(x, q=Q):
    x = np.atleast_1d(x).astype(np.int64)
    return x % q


def secure_reduction(x, lam, q=Q):
    x = np.atleast_1d(x).astype(np.int64)
    mask = np.random.rand(x.shape[-1] if x.ndim else 1) < lam
    central_vals = central_reduction(x, q)
    noncentral_vals = noncentral_reduction(x, q)
    return np.where(mask, central_vals, noncentral_vals)


# ============================================================================
# LEAKAGE MODEL
# ============================================================================

def leakage_with_cross_term(x, cross_term, noise_std=NOISE_STD, beta=BETA):
    hw = hamming_weight(x)
    noise = np.random.normal(0, noise_std, size=np.shape(hw))
    return hw + noise + beta * cross_term


# ============================================================================
# TRACE GENERATION (vectorized over the trace axis)
# ============================================================================

def generate_traces(secret, mode, lam=0.5, num_traces=NUM_TRACES):
    c = np.random.randint(1, Q, size=num_traces)
    s0 = np.random.randint(0, Q, size=num_traces)
    s1 = (secret - s0) % Q

    p0 = (s0 * c) % Q
    p1 = (s1 * c) % Q

    if mode == 'central':
        r0 = central_reduction(p0)
        r1 = central_reduction(p1)
    elif mode == 'noncentral':
        r0 = noncentral_reduction(p0)
        r1 = noncentral_reduction(p1)
    elif mode in ('secure', 'adaptive'):
        r0 = secure_reduction(p0, lam)
        r1 = secure_reduction(p1, lam)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cross = (r0.astype(np.float64) * r1.astype(np.float64))
    l0 = leakage_with_cross_term(r0, cross)
    l1 = leakage_with_cross_term(r1, cross)

    return l0, l1, c


# ============================================================================
# HOCPA ATTACK (fully vectorized over the 256 key guesses)
# ============================================================================

def hocpa_attack(L0, L1, C, key_space=KEY_SPACE):
    L0c = L0 - np.mean(L0)
    L1c = L1 - np.mean(L1)
    combined = L0c * L1c
    combined_centered = combined - np.mean(combined)

    guesses = np.arange(key_space)[:, None]          # (256, 1)
    prod = (guesses * C[None, :]) % Q                # (256, N)
    red = central_reduction(prod.reshape(-1)).reshape(prod.shape)
    pred = red.astype(np.float64) ** 2
    pred_centered = pred - pred.mean(axis=1, keepdims=True)

    num = pred_centered @ combined_centered
    denom = (np.sqrt((pred_centered ** 2).sum(axis=1)) *
             np.sqrt((combined_centered ** 2).sum()))
    correlations = np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-9)
    correlations = np.abs(correlations)

    best_guess = int(np.argmax(correlations))
    return best_guess, correlations


# ============================================================================
# ADAPTIVE CONTROLLER
# ============================================================================

class BoundedAdaptiveController:
    def __init__(self):
        self.lam = LAMBDA_INIT
        self.lambda_max = LAMBDA_MAX
        self.lambda_min = LAMBDA_MIN
        self.threshold = ADAPTIVE_THRESHOLD
        self.step_up = ADAPTIVE_STEP_UP
        self.step_down = ADAPTIVE_STEP_DOWN
        self.history = [LAMBDA_INIT]
        self.consecutive_failures = 0

    def update(self, peak_correlation, attack_success):
        if peak_correlation > self.threshold and attack_success:
            self.lam = min(self.lambda_max, self.lam + self.step_up)
            self.consecutive_failures = 0
        elif peak_correlation < self.threshold * 0.6 or not attack_success:
            self.consecutive_failures += 1
            if self.consecutive_failures > 2:
                self.lam = max(self.lambda_min, self.lam - self.step_down)
                self.consecutive_failures = 0
        else:
            self.consecutive_failures = 0
        self.history.append(self.lam)

    def get_lambda(self):
        return self.lam


# ============================================================================
# EXPERIMENT 1: ATTACK SUCCESS RATE SWEEP
# ============================================================================

def run_experiment(mode, fixed_lambda=None, adaptive=False):
    print(f"  Running {mode}...")
    trace_counts = list(range(TRACE_STEP, NUM_TRACES + 1, TRACE_STEP))
    success_rates = []
    peak_corr_means = []

    controller = BoundedAdaptiveController() if adaptive else None
    lam_current = fixed_lambda if fixed_lambda is not None else LAMBDA_FIXED

    for t in trace_counts:
        successes = 0
        peak_correlations = []

        for trial in range(NUM_TRIALS):
            secret = np.random.randint(0, KEY_SPACE)
            run_mode = 'secure' if adaptive else mode
            L0, L1, C = generate_traces(secret, run_mode, lam_current, t)
            best_guess, correlations = hocpa_attack(L0, L1, C)
            peak_corr = float(np.max(correlations))
            peak_correlations.append(peak_corr)

            if best_guess == secret:
                successes += 1

            if adaptive and controller and trial < NUM_TRIALS - 1:
                controller.update(peak_corr, best_guess == secret)
                lam_current = controller.get_lambda()

        success_rates.append(successes / NUM_TRIALS)
        peak_corr_means.append(float(np.mean(peak_correlations)))
        if adaptive:
            print(f"    {t} traces: success={success_rates[-1]:.2f}  "
                  f"mean r_max={peak_corr_means[-1]:.4f}  (λ={lam_current:.2f})")
        else:
            print(f"    {t} traces: success={success_rates[-1]:.2f}  "
                  f"mean r_max={peak_corr_means[-1]:.4f}")

    lambda_history = controller.history if adaptive else None
    return trace_counts, success_rates, peak_corr_means, lambda_history


# ============================================================================
# EXPERIMENT 2: SINGLE-SHOT r_max AT N=5000 (for the headline metric)
# ============================================================================

def measure_rmax(mode, lam=None, num_traces=NUM_TRACES_RMAX, num_repeats=10):
    peaks = []
    for _ in range(num_repeats):
        secret = np.random.randint(0, KEY_SPACE)
        L0, L1, C = generate_traces(secret, mode, lam if lam is not None else LAMBDA_FIXED, num_traces)
        _, correlations = hocpa_attack(L0, L1, C)
        peaks.append(float(np.max(correlations)))
    return float(np.mean(peaks)), float(np.std(peaks))


# ============================================================================
# EXPERIMENT 3: FUNCTIONAL CORRECTNESS
# ============================================================================

def check_correctness(num_samples=200000):
    x = np.random.randint(-5 * Q, 5 * Q, size=num_samples)
    ref = x % Q

    c = central_reduction(x) % Q
    n = noncentral_reduction(x) % Q
    s = secure_reduction(x, LAMBDA_FIXED) % Q

    correct_c = int(np.sum(c == ref))
    correct_n = int(np.sum(n == ref))
    correct_s = int(np.sum(s == ref))

    return {
        'total': num_samples,
        'central_pct': 100.0 * correct_c / num_samples,
        'noncentral_pct': 100.0 * correct_n / num_samples,
        'secure_pct': 100.0 * correct_s / num_samples,
    }


# ============================================================================
# EXPERIMENT 4: COMPUTATIONAL OVERHEAD
# ============================================================================

def measure_overhead(num_values=1_000_000, repeats=20):
    x = np.random.randint(0, 5 * Q, size=num_values)

    t0 = time.perf_counter()
    for _ in range(repeats):
        central_reduction(x)
    t_central = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        secure_reduction(x, LAMBDA_FIXED)
    t_secure = (time.perf_counter() - t0) / repeats

    overhead_pct = 100.0 * (t_secure - t_central) / t_central
    return t_central, t_secure, overhead_pct


# ============================================================================
# EXPERIMENT 5: SIGNAL-TO-NOISE RATIO
# ============================================================================

def measure_snr(mode, lam=None, num_traces=50000):
    np.random.seed(123)
    c = np.random.randint(1, Q, size=num_traces)
    secret = np.random.randint(0, KEY_SPACE)
    s0 = np.random.randint(0, Q, size=num_traces)
    p0 = (s0 * c) % Q
    if mode == 'central':
        r0 = central_reduction(p0)
    elif mode == 'noncentral':
        r0 = noncentral_reduction(p0)
    else:
        r0 = secure_reduction(p0, lam if lam is not None else LAMBDA_FIXED)
    signal = hamming_weight(r0).astype(np.float64)
    var_signal = float(np.var(signal))
    var_noise = float(NOISE_STD ** 2)
    np.random.seed(42)
    return var_signal / var_noise, var_signal, var_noise


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(central_results, noncentral_results, secure_results, adaptive_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    central_x, central_y, _, _ = central_results
    noncentral_x, noncentral_y, _, _ = noncentral_results
    secure_x, secure_y, _, _ = secure_results
    adaptive_x, adaptive_y, _, lambda_hist = adaptive_results

    ax1 = axes[0]
    ax1.plot(central_x, central_y, 'o-', color='#E74C3C', linewidth=2.5,
              markersize=6, label='No Protection (Central)', alpha=0.6)
    ax1.plot(noncentral_x, noncentral_y, 's-', color='#3498DB', linewidth=2,
              markersize=6, label='Non-Central Reduction', alpha=0.7)
    ax1.plot(secure_x, secure_y, '^-', color='#2ECC71', linewidth=2.5,
              markersize=8, label='Fixed Secure (λ = 0.5)', alpha=0.9)
    ax1.plot(adaptive_x, adaptive_y, 'D-', color='#F39C12', linewidth=3,
              markersize=8, label='Bounded Adaptive (λ ≤ 0.7)', alpha=1.0)
    ax1.set_xlabel('Number of Power Traces', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attack Success Rate', fontsize=12, fontweight='bold')
    ax1.set_title('(a) HOCPA Attack Success Rate Comparison (lower = more secure)',
                   fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])

    ax2 = axes[1]
    if lambda_hist:
        step = max(1, len(lambda_hist) // len(adaptive_x))
        lambda_at_points = lambda_hist[::step][:len(adaptive_x)]
        ax2.plot(adaptive_x, lambda_at_points, 'D-', color='#F39C12',
                  linewidth=2.5, markersize=7, label='Adaptive λ(t)')
        ax2.axhline(y=0.5, color='#2ECC71', linestyle='--', linewidth=2,
                     label='Fixed Secure Baseline (λ = 0.5)')
        ax2.axhline(y=0.3, color='gray', linestyle=':', linewidth=1.5,
                     label='Initial λ = 0.3', alpha=0.7)
        ax2.axhline(y=0.7, color='red', linestyle=':', linewidth=2,
                     label='Upper Bound (λ ≤ 0.7)', alpha=0.8)
        ax2.set_xlabel('Number of Power Traces', fontsize=12, fontweight='bold')
        ax2.set_ylabel('λ (Randomization Probability)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Bounded Adaptive λ Response', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0.15, 0.75])

    plt.tight_layout()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("HIGHER-ORDER CORRELATION POWER ANALYSIS (HOCPA)")
    print("Bounded Adaptive Central Reduction for Masked Kyber -- v2")
    print("=" * 80)

    print("\n[1/4] Baseline: Central Reduction")
    central_results = run_experiment('central')
    print("\n[2/4] Baseline: Non-Central Reduction")
    noncentral_results = run_experiment('noncentral')
    print("\n[3/4] Countermeasure: Fixed Secure Reduction (λ = 0.5)")
    secure_results = run_experiment('secure', fixed_lambda=LAMBDA_FIXED)
    print("\n[4/4] Countermeasure: Bounded Adaptive Secure Reduction")
    adaptive_results = run_experiment('adaptive', adaptive=True)

    plot_results(central_results, noncentral_results, secure_results, adaptive_results)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2: PEAK CORRELATION r_max AT N={NUM_TRACES_RMAX}")
    print("=" * 80)
    rmax_central, std_central = measure_rmax('central')
    rmax_noncentral, std_noncentral = measure_rmax('noncentral')
    rmax_secure, std_secure = measure_rmax('secure', LAMBDA_FIXED)
    rmax_adaptive, std_adaptive = measure_rmax('secure', 0.7)
    print(f"  Central:      r_max = {rmax_central:.4f} (± {std_central:.4f})")
    print(f"  Non-central:  r_max = {rmax_noncentral:.4f} (± {std_noncentral:.4f})")
    print(f"  Fixed secure: r_max = {rmax_secure:.4f} (± {std_secure:.4f})")
    print(f"  Adaptive:     r_max = {rmax_adaptive:.4f} (± {std_adaptive:.4f})")

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: FUNCTIONAL CORRECTNESS")
    print("=" * 80)
    print(check_correctness())

    print("\n" + "=" * 80)
    print("EXPERIMENT 4: COMPUTATIONAL OVERHEAD")
    print("=" * 80)
    t_central, t_secure, overhead_pct = measure_overhead()
    print(f"  Central reduction:  {t_central*1e3:.4f} ms / 1M values")
    print(f"  Secure reduction:   {t_secure*1e3:.4f} ms / 1M values")
    print(f"  Overhead:           {overhead_pct:.2f}%")

    print("\n" + "=" * 80)
    print("EXPERIMENT 5: SIGNAL-TO-NOISE RATIO")
    print("=" * 80)
    snr_central, vs_c, vn_c = measure_snr('central')
    snr_secure, vs_s, vn_s = measure_snr('secure', LAMBDA_FIXED)
    print(f"  Central: SNR = {snr_central:.4f}  (Var(signal)={vs_c:.4f}, Var(noise)={vn_c:.4f})")
    print(f"  Secure:  SNR = {snr_secure:.4f}  (Var(signal)={vs_s:.4f}, Var(noise)={vn_s:.4f})")

    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE (Attack Success Rate @ 5000 traces)")
    print("=" * 80)
    central_rate = central_results[1][-1]
    noncentral_rate = noncentral_results[1][-1]
    secure_rate = secure_results[1][-1]
    adaptive_rate = adaptive_results[1][-1]
    print(f"\n{'Protection Mode':<35} {'Success Rate':<15}")
    print("-" * 50)
    print(f"{'No Protection (Central)':<35} {central_rate:.1%}")
    print(f"{'Non-Central Reduction':<35} {noncentral_rate:.1%}")
    print(f"{'Fixed Secure (λ = 0.5)':<35} {secure_rate:.1%}")
    print(f"{'Bounded Adaptive (λ ≤ 0.7)':<35} {adaptive_rate:.1%}")

    reduction = central_rate - adaptive_rate
    print(f"\n{'='*80}")
    print(f"Adaptive reduces attacker success rate by {reduction:.1%} "
          f"relative to the unprotected baseline (lower is better for defense).")
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
