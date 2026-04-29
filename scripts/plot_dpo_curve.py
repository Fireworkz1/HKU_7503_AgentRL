import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps, losses = [], []
with open("dpo_train_v2.log", "r", encoding="utf-8") as f:
    for line in f:
        m = re.search(r"Epoch:\[(\d+)/2\]\((\d+)/4292\), loss: ([\d.]+)", line)
        if m:
            epoch = int(m.group(1))
            step = int(m.group(2))
            global_step = (epoch - 1) * 4292 + step
            steps.append(global_step)
            losses.append(float(m.group(3)))

steps = np.array(steps)
losses = np.array(losses)

def smooth(y, window=5):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(steps, losses, color="#2196F3", alpha=0.12, linewidth=0.6)

w1 = 10
sm1 = smooth(losses, w1)
ax.plot(steps[w1-1:], sm1, color="#64B5F6", linewidth=1.2, alpha=0.5, label="DPO Loss (window=10)")

w2 = 20
sm2 = smooth(losses, w2)
ax.plot(steps[w2-1:], sm2, color="#1565C0", linewidth=3, label="DPO Loss (window=20)")

ax.axhline(y=np.log(2), color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(max(steps)*0.02, np.log(2)+0.03, f'random baseline (ln2≈0.693)', fontsize=10, color='gray', alpha=0.8)

ax.axvline(x=4292, color='#E53935', linestyle=':', linewidth=1.5, alpha=0.6)
ax.text(4292+50, max(losses)*0.9, 'Epoch 1→2', fontsize=10, color='#E53935', alpha=0.8)

first_avg = np.mean(losses[:10])
last_avg = np.mean(losses[-10:])
ax.annotate(f'Start: {first_avg:.3f}', xy=(steps[5], first_avg), fontsize=11,
            xytext=(steps[5]+400, first_avg+0.25),
            arrowprops=dict(arrowstyle='->', color='#333'), color='#333', fontweight='bold')
ax.annotate(f'End: {last_avg:.3f}', xy=(steps[-5], last_avg), fontsize=11,
            xytext=(steps[-5]-1500, last_avg+0.35),
            arrowprops=dict(arrowstyle='->', color='#333'), color='#333', fontweight='bold')

ax.set_xlabel("Training Step (global)", fontsize=13)
ax.set_ylabel("DPO Loss", fontsize=13)
ax.set_title("DPO Training Curve v2  (MiniMind 64M · 2 epochs · lr=5e-7)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.set_xlim(0, max(steps) * 1.02)
ax.set_ylim(0, max(losses) * 1.05)
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("d:/minimind_rl_experiment/charts/dpo_training_curve.png", dpi=150)
fig.savefig("d:/minimind/eval_results/dpo_training_curve.png", dpi=150)
plt.close(fig)
print(f"Done. Steps: {len(steps)}, Loss: {losses.min():.4f}-{losses.max():.4f}")
print(f"First 10 avg: {first_avg:.4f}, Last 10 avg: {last_avg:.4f}, Delta: {first_avg - last_avg:+.4f}")
