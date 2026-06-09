from pathlib import Path
import re
import shutil

p = Path("main_train.py")
bak = Path("main_train.py.bak_before_weak3_save")
if not bak.exists():
    shutil.copy2(p, bak)
    print(f"[backup] {p} -> {bak}")

s = p.read_text(encoding="utf-8")

# 1. add weak3_f1 to choices
s = s.replace(
    "choices=['loss', 'eer', 'f1', 'safe_f1']",
    "choices=['loss', 'eer', 'f1', 'safe_f1', 'weak3_f1']"
)

s = s.replace(
    'choices=[\'loss\', \'eer\', \'f1\']',
    'choices=[\'loss\', \'eer\', \'f1\', \'weak3_f1\']'
)

# 2. add prev_weak3_f1
if "prev_weak3_f1" not in s:
    s = s.replace(
        'prev_f1 = -float("inf")',
        'prev_f1 = -float("inf")\n    prev_weak3_f1 = -float("inf")',
        1
    )

# 3. compute weak3_f1 after type_f1s
if "val_weak3_f1" not in s:
    old = '''                print("Track2 Type F1s [speech, sound, singing, music]:", type_f1s)'''
    new = '''                print("Track2 Type F1s [speech, sound, singing, music]:", type_f1s)

                if len(type_f1s) >= 4:
                    val_weak3_f1 = float(np.mean([type_f1s[0], type_f1s[1], type_f1s[3]]))
                else:
                    val_weak3_f1 = val_f1

                print("Val Weak3F1 [speech/sound/music]: {}".format(val_weak3_f1))'''
    if old not in s:
        raise RuntimeError("Cannot find Track2 Type F1s print block.")
    s = s.replace(old, new, 1)

# 4. make weak3_f1 available in non-track2 path
if "if 'val_weak3_f1' not in locals():" not in s:
    marker = '            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:'
    insert = '''            if 'val_weak3_f1' not in locals():
                val_weak3_f1 = val_f1

'''
    if marker not in s:
        raise RuntimeError("Cannot find dev_loss.log marker.")
    s = s.replace(marker, insert + marker, 1)

# 5. add save_best_by branch
old = '''        elif args.save_best_by == "safe_f1":
            if t2_safe_f1 > prev_safe_f1:
                prev_safe_f1 = t2_safe_f1
                save_flag = True'''
new = '''        elif args.save_best_by == "safe_f1":
            if t2_safe_f1 > prev_safe_f1:
                prev_safe_f1 = t2_safe_f1
                save_flag = True

        elif args.save_best_by == "weak3_f1":
            if val_weak3_f1 > prev_weak3_f1:
                prev_weak3_f1 = val_weak3_f1
                save_flag = True'''
if old in s:
    s = s.replace(old, new, 1)
else:
    old2 = '''        elif args.save_best_by == "f1":
            if val_f1 > prev_f1:
                prev_f1 = val_f1
                save_flag = True'''
    new2 = '''        elif args.save_best_by == "f1":
            if val_f1 > prev_f1:
                prev_f1 = val_f1
                save_flag = True

        elif args.save_best_by == "weak3_f1":
            if val_weak3_f1 > prev_weak3_f1:
                prev_weak3_f1 = val_weak3_f1
                save_flag = True'''
    if old2 not in s:
        raise RuntimeError("Cannot find save_best_by branch.")
    s = s.replace(old2, new2, 1)

p.write_text(s, encoding="utf-8")
print("[done] added weak3_f1 checkpoint selection")
