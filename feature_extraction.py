import os
import re
import time
import numpy as np
import pandas as pd
import librosa
from spafe.features.rplp import plp
from multiprocessing import Pool, cpu_count

SKIPPED_FILE = "skipped_files.txt"
COMBINED_CSV = "combined_features.csv"


def process_file(args):
    speaker, file_path = args

    try:
        audio, sr = librosa.load(file_path, sr=16000, res_type="kaiser_fast")
        audio = audio + np.random.normal(0, 1e-6, size=audio.shape)

        # ---------------- MFCC ----------------
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # ---------------- LPC ----------------
        frame_length = 2048
        hop_length = 1024
        lpc_order = 12
        lpc_features = []

        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            try:
                coeffs = librosa.lpc(frame, order=lpc_order)
                lpc_features.append(coeffs)
            except Exception:
                continue

        if len(lpc_features) == 0:
            lpc_mean = np.zeros(lpc_order + 1)
        else:
            lpc_mean = np.mean(lpc_features, axis=0)

        # ---------------- PLP ----------------
        try:
            plp_features = plp(audio, fs=sr, order=13)
            plp_mean = np.mean(plp_features, axis=0)
        except Exception:
            plp_mean = np.zeros(13)

        combined = np.concatenate([mfcc_mean, lpc_mean, plp_mean])
        row = list(combined) + [speaker]
        return row, None

    except Exception as e:
        return None, f"{file_path} -> {e}"


def parse_skipped_file(path):
    """Extract just the file paths from skipped_files.txt, ignoring traceback noise."""
    jobs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            file_path = line.split(" -> ")[0].strip()
            if not file_path.lower().endswith(".wav"):
                continue
            speaker = os.path.basename(os.path.dirname(file_path))
            jobs.append((speaker, file_path))
    return jobs


def main():
    start_time = time.time()

    jobs = parse_skipped_file(SKIPPED_FILE)
    jobs = list(dict.fromkeys(jobs))  # de-duplicate

    print(f"Found {len(jobs)} skipped files to reprocess")
    print(f"Using {cpu_count()} CPU cores")

    new_rows = []
    still_skipped = []

    with Pool(processes=cpu_count()) as pool:
        for i, (row, error) in enumerate(pool.imap_unordered(process_file, jobs, chunksize=4), 1):
            if row is not None:
                new_rows.append(row)
            else:
                still_skipped.append(error)

            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time() - start_time
                print(f"Processed {i}/{len(jobs)} files... ({elapsed:.1f}s elapsed)")

    columns = (
        [f"MFCC_{i+1}" for i in range(13)] +
        [f"LPC_{i+1}" for i in range(13)] +
        [f"PLP_{i+1}" for i in range(13)] +
        ["Speaker"]
    )

    new_df = pd.DataFrame(new_rows, columns=columns)

    if os.path.exists(COMBINED_CSV):
        old_df = pd.read_csv(COMBINED_CSV)
        merged_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged_df = new_df

    merged_df.to_csv(COMBINED_CSV, index=False)

    total_time = time.time() - start_time
    print(f"\nReprocessing completed in {total_time:.1f} seconds!")
    print(f"Newly recovered rows: {len(new_df)}")
    print(f"Combined dataset shape (after merge): {merged_df.shape}")
    print(f"Still skipped (genuine errors): {len(still_skipped)}")

    if still_skipped:
        with open("still_skipped_files.txt", "w") as f:
            f.write("\n".join(still_skipped))
        print("List of remaining skipped files saved to still_skipped_files.txt")


if __name__ == "__main__":
    main()