"""Backblaze Drive Stats: download, verification, extraction and per-drive cohort records (protocol_v6.md, Part 5).

NOT RUN in this repository: the data hosts were unreachable from the computing environment (network policy). The code
was written against Backblaze's published schema and tested only on a small code fixture (test_field.py).

Data: quarterly ZIP files data_Q<q>_<year>.zip from https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/ ,
each holding one CSV per day with columns date, serial_number, model, capacity_bytes, failure, [datacenter, cluster_id,
vault_id, pod_id, pod_slot_num, is_legacy_format (2023+)], smart_<id>_normalized, smart_<id>_raw. Cite Backblaze as the
source. failure = 1 on a drive's last day of operation before it failed; drives removed for other reasons stop
reporting (censored).

Usage:
    python -m field.backblaze fetch 2022 2024          # download 2022Q1..2024Q4 into data/field/backblaze/raw, SHA-256 manifest
    python -m field.backblaze extract                  # CSV -> Parquet (selected columns), one file per quarter
    python -m field.backblaze describe 2022            # cohort description (drives, failures, censored, by model)
"""
import glob, hashlib, io, json, os, shutil, sys, tempfile, time, urllib.request, zipfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, 'data', 'field', 'backblaze')
RAW, PQ = os.path.join(BASE, 'raw'), os.path.join(BASE, 'parquet')
URL = 'https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q{q}_{y}.zip'
SMART = [5, 9, 187, 188, 194, 197, 198, 1, 7, 10, 12, 199]           # protocol_v6: primary list, then "when present"
KEEP = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure'] + [f'smart_{i}_raw' for i in SMART]
MIN_DRIVES = 5000                                                       # models with >= 5,000 drives in the 2022 cohort


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()


def fetch(y0, y1, retries=4):
    os.makedirs(RAW, exist_ok=True); man = {}
    for y in range(y0, y1 + 1):
        for q in (1, 2, 3, 4):
            url = URL.format(q=q, y=y); dst = os.path.join(RAW, os.path.basename(url))
            if not os.path.exists(dst):
                for k in range(retries):
                    try:
                        urllib.request.urlretrieve(url, dst + '.part'); os.replace(dst + '.part', dst); break
                    except Exception as e:                      # network errors: exponential backoff
                        print('retry', url, e); time.sleep(2 ** (k + 1))
            if os.path.exists(dst):
                man[os.path.basename(dst)] = dict(sha256=sha256(dst), bytes=os.path.getsize(dst), url=url)
    json.dump(man, open(os.path.join(RAW, 'MANIFEST.json'), 'w'), indent=1)
    return man


def extract(zips=None):
    """one Parquet file per quarter with the KEEP columns present in that quarter (missing ones as NULL)"""
    import duckdb
    os.makedirs(PQ, exist_ok=True)
    for z in sorted(zips or glob.glob(os.path.join(RAW, 'data_Q*_*.zip'))):
        out = os.path.join(PQ, os.path.basename(z).replace('.zip', '.parquet'))
        if os.path.exists(out): continue
        tmp = tempfile.mkdtemp(dir=BASE)
        try:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.endswith('.csv') and '__MACOSX' not in n]
                for n in names: zf.extract(n, tmp)
            files = sorted(glob.glob(os.path.join(tmp, '**', '*.csv'), recursive=True))
            con = duckdb.connect()
            cols = set(pd.read_csv(files[-1], nrows=0).columns) | set(pd.read_csv(files[0], nrows=0).columns)
            sel = ', '.join([f'"{c}"' if c in cols else f'NULL AS "{c}"' for c in KEEP])
            flist = '[' + ', '.join(f"'{f}'" for f in files) + ']'
            con.execute(f"COPY (SELECT {sel} FROM read_csv({flist}, union_by_name=true, header=true, all_varchar=true)) "
                        f"TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
            print(os.path.basename(out), n, 'rows from', len(files), 'daily files', flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def load_year(year, models=None):
    """daily rows of one calendar year (typed), optionally restricted to models"""
    import duckdb
    files = sorted(glob.glob(os.path.join(PQ, f'data_Q*_{year}.parquet')))
    flist = '[' + ', '.join(f"'{f}'" for f in files) + ']'
    smart = ', '.join(f'TRY_CAST(smart_{i}_raw AS DOUBLE) AS s{i}' for i in SMART)
    where = '' if models is None else 'WHERE model IN (' + ', '.join(f"'{m}'" for m in models) + ')'
    q = (f"SELECT CAST(date AS DATE) AS date, serial_number, model, TRY_CAST(capacity_bytes AS DOUBLE) AS capacity, "
         f"TRY_CAST(failure AS INTEGER) AS failure, {smart} FROM read_parquet({flist}) {where}")
    return duckdb.connect().execute(q).df()


def cohort_units(D, year):
    """per-drive cohort table for year Y: drives reporting on 1 January; failure = first failure flag in the year;
    censoring = last report before 31 December without a failure record; drives reporting after a failure record are
    excluded (returned separately for the count)."""
    d0, d1 = pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31')
    D = D[(D.date >= d0) & (D.date <= d1)]
    first = D.groupby('serial_number').date.min()
    cohort = first.index[first == d0]
    D = D[D.serial_number.isin(cohort)].sort_values(['serial_number', 'date'])
    g = D.groupby('serial_number')
    fail_date = D[D.failure == 1].groupby('serial_number').date.min()
    last = g.date.max()
    U = pd.DataFrame(dict(model=g.model.first(), capacity=g.capacity.first(), last=last, age0=g.s9.first() / 24.0))
    U['fail_date'] = fail_date.reindex(U.index)
    U['after_failure'] = U.fail_date.notna() & (U['last'] > U.fail_date)
    U['event'] = U.fail_date.notna()
    U['T'] = (U.fail_date - d0).dt.days + 1.0                         # failure during day T (rows t = 0 .. T-1)
    U['c'] = np.where(U.event, np.nan, (U['last'] - d0).dt.days + 1.0)   # last seen alive at the end of day c - 1
    U['censored_before_end'] = ~U.event & (U['last'] < d1)
    return U, D


def describe(year, models=None):
    D = load_year(year, models); U, _ = cohort_units(D, year)
    by = U.groupby('model').agg(drives=('event', 'size'), failures=('event', 'sum'), removed=('censored_before_end', 'sum'),
                                after_failure=('after_failure', 'sum')).sort_values('drives', ascending=False)
    return by


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'fetch': print(json.dumps(fetch(int(sys.argv[2]), int(sys.argv[3])), indent=1))
    elif cmd == 'extract': extract()
    elif cmd == 'describe': print(describe(int(sys.argv[2])).to_string())
