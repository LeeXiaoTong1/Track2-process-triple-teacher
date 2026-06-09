from pathlib import Path
import re
import shutil

p = Path("main_train.py")
bak = Path("main_train.py.bak_before_singing_safe_save")
if not bak.exists():
    shutil.copy2(p, bak)
    print(f"[backup] {p} -> {bak}")

s = p.read_text(encoding="utf-8")

# 1. Add safe_f1 choice.
s = re.sub(
    r"choices=\[['\"]loss['\"],\s*['\"]eer['\"],\s*['\"]f1['\"]\]",
    "choices=['loss', 'eer', 'f1', 'safe_f1']",
    s,
    count=1
)

# 2. Add arguments.
if "--t2_singing_floor" not in s:
    marker = "    # generalized strategy"
    insert = r'''
    # Track2 Singing-safe checkpoint selection.
    parser.add_argument(
        '--t2_singing_floor',
        type=float,
        default=0.94,
        help='Singing F1 floor used by safe_f1 checkpoint selection.'
    )
    parser.add_argument(
        '--t2_singing_penalty',
        type=float,
        default=1.5,
        help='Penalty strength when Singing F1 is below t2_singing_floor.'
    )
'''
    if marker not in s:
        raise RuntimeError("Cannot find marker: # generalized strategy")
    s = s.replace(marker, insert + "\n" + marker, 1)

# 3. Better dev log header.
s = s.replace(
    'file.write("epoch\\tval_loss\\tval_eer\\tval_f1\\n")',
    'file.write("epoch\\tval_loss\\tval_eer\\tval_f1\\tval_safe_f1\\n")',
    1
)

# 4. Add prev_safe_f1.
if "prev_safe_f1" not in s:
    s = s.replace(
        'prev_f1 = -float("inf")',
        'prev_f1 = -float("inf")\n    prev_safe_f1 = -float("inf")',
        1
    )

# 5. Insert safe_f1 computation before dev log write.
if "t2_safe_f1 = val_f1" not in s:
    marker = '            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:'
    insert = r'''            t2_safe_f1 = val_f1
            singing_f1 = None

            if (
                args.train_task == "atadd-track2"
                and len(type_loader) > 0
                and "type_f1s" in locals()
                and len(type_f1s) > 2
            ):
                singing_f1 = float(type_f1s[2])
                singing_gap = max(0.0, float(args.t2_singing_floor) - singing_f1)
                t2_safe_f1 = float(val_f1) - float(args.t2_singing_penalty) * singing_gap

            print("Val SafeF1: {}".format(t2_safe_f1))
            if singing_f1 is not None:
                print("Singing F1 guard: singing_f1={}, floor={}, penalty={}".format(
                    singing_f1,
                    args.t2_singing_floor,
                    args.t2_singing_penalty
                ))

'''
    if marker not in s:
        raise RuntimeError("Cannot find dev_loss.log marker")
    s = s.replace(marker, insert + marker, 1)

# 6. Add safe_f1 into log line.
s = s.replace(
    'str(val_f1) + "\\n"',
    'str(val_f1) + "\\t" + str(t2_safe_f1) + "\\n"',
    1
)

# 7. Add save_best_by safe_f1 branch.
old = r'''        elif args.save_best_by == "f1":
            if val_f1 > prev_f1:
                prev_f1 = val_f1
                save_flag = True'''
new = r'''        elif args.save_best_by == "f1":
            if val_f1 > prev_f1:
                prev_f1 = val_f1
                save_flag = True

        elif args.save_best_by == "safe_f1":
            if t2_safe_f1 > prev_safe_f1:
                prev_safe_f1 = t2_safe_f1
                save_flag = True'''
if old not in s:
    raise RuntimeError("Cannot find save_best_by f1 branch")
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("[done] added Singing-safe checkpoint selection")
