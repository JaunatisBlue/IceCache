"""Same-protocol comparison: baseline 142657a vs the current working tree.

Every number here comes from ``batch_decode_probe.py`` run with the same timing
protocol (steady-state decode = main phase with the first two steps dropped).
The baseline tree was rebuilt with ``git archive 142657a`` plus the built
extensions, so nothing was stashed or checked out in the shared working tree.
"""

import json


def load(path):
    return json.load(open(path))


def stats(path, key="decode"):
    return load(path)["step_stats"][key]


print("=== decode 同口径对照（稳态：主相位丢弃前 2 步）===")
print(f"{'B':>2} {'基线 ms/步':>12} {'现在 ms/步':>12} {'变化':>8} "
      f"{'基线 tok/s':>11} {'现在 tok/s':>11}")
base_per, now_per = {}, {}
for bs in (1, 2, 8):
    b = stats(f"/tmp/base_b{bs}.json")["mean"]
    n = stats(f"/tmp/t_b{bs}.json")["mean"]
    base_per[bs], now_per[bs] = b / bs, n / bs
    print(f"{bs:>2} {b:12.2f} {n:12.2f} {100 * (n - b) / b:7.1f}% "
          f"{1000 * bs / b:11.2f} {1000 * bs / n:11.2f}")

print()
print(f"批量扩展性   : 基线 {base_per[1] / base_per[8]:.2f}x  ->  现在 {now_per[1] / now_per[8]:.2f}x")
print(f"B=8 每 token : 基线 {base_per[8]:.2f} ms  ->  现在 {now_per[8]:.2f} ms  "
      f"({100 * (now_per[8] - base_per[8]) / base_per[8]:+.1f}%)")
print(f"B=8 吞吐     : 基线 {1000 / base_per[8]:.2f}  ->  现在 {1000 / now_per[8]:.2f} tok/s")

print()
print("--- 旧口径（含冷启动的全 8 步均值），用来说明口径差异有多大 ---")
for bs in (1, 2, 8):
    b = stats(f"/tmp/base_b{bs}.json", "decode_including_cold")["mean"]
    n = stats(f"/tmp/t_b{bs}.json", "decode_including_cold")["mean"]
    print(f"  B={bs}: 基线 {b:7.2f} -> 现在 {n:7.2f} ms ({100 * (n - b) / b:+.1f}%) | "
          f"稳态 {stats(f'/tmp/t_b{bs}.json')['mean']:7.2f} ms")

print()
print("=== prefill 同口径（skewed：slot0 = 4096 token，其余 1024-1136）===")
for tag, path in (("基线 整批", "/tmp/base_skew_b.json"),
                  ("现在 整批", "/tmp/q_s_gather.json"),
                  ("现在 长度分组 8192", "/tmp/q_s_group.json"),
                  ("基线 串行", "/tmp/base_skew_s.json"),
                  ("现在 串行", "/tmp/q_s_seq.json")):
    d = load(path)
    print(f"  {tag:18s} {sum(d['prefill_seconds']):6.3f} s   "
          f"groups={d.get('prefill_groups')} padded_rows={d.get('prefill_padded_rows')}")

print()
print("=== uniform prefill 同口径（1k x 8，B=8 那一次）===")
for tag, path in (("基线 整批", "/tmp/base_b8.json"),
                  ("现在 整批", "/tmp/t_b8.json")):
    d = load(path)
    print(f"  {tag:18s} {sum(d['prefill_seconds']):6.3f} s   "
          f"groups={d.get('prefill_groups')} padded_rows={d.get('prefill_padded_rows')}")
