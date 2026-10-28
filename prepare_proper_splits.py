import os, shutil, random
from pathlib import Path

# Fixed seed for 100% reproducible, non-leaking data split
SEED = 42
random.seed(SEED)

FACTORIES = {
    'factory_a': 'metal_nut',
    'factory_b': 'transistor',
    'factory_c': 'carpet'
}

BASE_DATA = Path('data')

split_summary = {}

for fid, cat in FACTORIES.items():
    raw_root = BASE_DATA / fid / cat
    out_root = BASE_DATA / fid / "clean_split"

    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)

    for split in ['train', 'val', 'test']:
        (out_root / split / 'good').mkdir(parents=True, exist_ok=True)
        (out_root / split / 'defect').mkdir(parents=True, exist_ok=True)

    train_files = set()
    val_files = set()
    test_files = set()

    # 1. Stratified Defect Split (30% test, 20% val, 50% train per defect subtype)
    defect_counts = {'train': 0, 'val': 0, 'test': 0}
    for sub in (raw_root / 'test').iterdir():
        if not sub.is_dir() or sub.name == 'good':
            continue
        defect_type = sub.name
        imgs = sorted(list(sub.glob('*.png')) + list(sub.glob('*.jpg')))
        random.shuffle(imgs)

        n = len(imgs)
        n_test = max(1, int(round(n * 0.30)))
        n_val = max(1, int(round(n * 0.20)))
        n_train = n - n_test - n_val
        if n_train <= 0:
            n_train = 1
            if n_val > 1: n_val -= 1
            elif n_test > 1: n_test -= 1

        test_imgs = imgs[:n_test]
        val_imgs = imgs[n_test:n_test + n_val]
        train_imgs = imgs[n_test + n_val:]

        for img in train_imgs:
            dst_name = f"{defect_type}_{img.name}"
            shutil.copy2(img, out_root / 'train' / 'defect' / dst_name)
            train_files.add(str(img.resolve()))
            defect_counts['train'] += 1

        for img in val_imgs:
            dst_name = f"{defect_type}_{img.name}"
            shutil.copy2(img, out_root / 'val' / 'defect' / dst_name)
            val_files.add(str(img.resolve()))
            defect_counts['val'] += 1

        for img in test_imgs:
            dst_name = f"{defect_type}_{img.name}"
            shutil.copy2(img, out_root / 'test' / 'defect' / dst_name)
            test_files.add(str(img.resolve()))
            defect_counts['test'] += 1

    # 2. Good Image Split (Test-Good into Test/Val, Train-Good into Train/Val)
    test_goods = sorted(list((raw_root / 'test' / 'good').glob('*.png')))
    random.shuffle(test_goods)
    
    # 30% of test goods into test
    n_tg = len(test_goods)
    n_test_g = max(1, int(round(n_tg * 0.50)))
    test_g_imgs = test_goods[:n_test_g]
    val_g_from_test = test_goods[n_test_g:]

    for img in test_g_imgs:
        shutil.copy2(img, out_root / 'test' / 'good' / f"test_{img.name}")
        test_files.add(str(img.resolve()))

    for img in val_g_from_test:
        shutil.copy2(img, out_root / 'val' / 'good' / f"test_{img.name}")
        val_files.add(str(img.resolve()))

    # Train goods
    train_goods = sorted(list((raw_root / 'train' / 'good').glob('*.png')))
    random.shuffle(train_goods)

    # Put 20% into val to have a solid validation set, rest into train
    n_trg = len(train_goods)
    n_val_trg = int(round(n_trg * 0.20))
    val_g_imgs = train_goods[:n_val_trg]
    train_g_imgs = train_goods[n_val_trg:]

    for img in val_g_imgs:
        shutil.copy2(img, out_root / 'val' / 'good' / f"train_{img.name}")
        val_files.add(str(img.resolve()))

    for img in train_g_imgs:
        shutil.copy2(img, out_root / 'train' / 'good' / f"train_{img.name}")
        train_files.add(str(img.resolve()))

    # Strict Zero-Leakage Invariance Verification
    assert len(train_files & val_files) == 0, f"LEAKAGE DETECTED between Train and Val in {fid}!"
    assert len(train_files & test_files) == 0, f"LEAKAGE DETECTED between Train and Test in {fid}!"
    assert len(val_files & test_files) == 0, f"LEAKAGE DETECTED between Val and Test in {fid}!"

    split_summary[fid] = {
        'train_good': len(train_g_imgs),
        'train_defect': defect_counts['train'],
        'val_good': len(val_g_imgs) + len(val_g_from_test),
        'val_defect': defect_counts['val'],
        'test_good': len(test_g_imgs),
        'test_defect': defect_counts['test']
    }

print("=== SPLIT GENERATION COMPLETE (ZERO DATA LEAKAGE) ===")
for fid, counts in split_summary.items():
    print(f"\n[{fid} - {FACTORIES[fid]}]")
    print(f"  Train: {counts['train_good']} good, {counts['train_defect']} defect (Total: {counts['train_good'] + counts['train_defect']})")
    print(f"  Val:   {counts['val_good']} good, {counts['val_defect']} defect (Total: {counts['val_good'] + counts['val_defect']})")
    print(f"  Test:  {counts['test_good']} good, {counts['test_defect']} defect (Total: {counts['test_good'] + counts['test_defect']})")
