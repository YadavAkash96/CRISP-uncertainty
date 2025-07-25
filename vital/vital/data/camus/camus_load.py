import h5py

def count_samples(h5_path: str, fold: str = "fold_1"):
    with h5py.File(h5_path, "r") as f:
        cross_val_group = f["cross_validation"][fold]

        def decode_keys(keys):
            return [k.decode("utf-8") for k in keys]

        train_patients = decode_keys(cross_val_group["train"][:])
        val_patients = decode_keys(cross_val_group["val"][:])
        test_patients = decode_keys(cross_val_group["test"][:])
        print(f"Patients per split in {fold}:")
        print(f"  Train: {len(train_patients)}")
        print(f"  Val  : {len(val_patients)}")
        print(f"  Test : {len(test_patients)}")

        def count_patient_samples(pids):
            sample_count = 0
            for pid in pids:
                if pid not in f:
                    print(f"Warning: Patient {pid} not found in HDF5 file.")
                    continue
                patient_group = f[pid]
                for view in patient_group.keys():
                    img_proc = patient_group[view]["img_proc"]
                    num_frames = img_proc.shape[0]  # sequence length or 2 (ED+ES)
                    sample_count += num_frames
            return sample_count

        train_val_patients = train_patients + val_patients
        total_train_val_samples = count_patient_samples(train_val_patients)
        total_test_samples = count_patient_samples(test_patients)
        total_all_samples = count_patient_samples(list(f.keys() - {"cross_validation"}))

        print("\nSample counts:")
        print(f"  Train + Val: {total_train_val_samples}")
        print(f"  Test       : {total_test_samples}")
        print(f"  All        : {total_all_samples}")

if __name__ == "__main__":
    path_to_h5 = "C:\Internship_work\CRISP-uncertainty\config\data\camus_h5\camus.h5"  # Change this to your actual path
    count_samples(path_to_h5)
