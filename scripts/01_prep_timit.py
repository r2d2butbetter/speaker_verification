"""Parse TIMIT and create background/target lists.

This script is a placeholder scaffold. Fill paths and logic as needed.
"""

from pathlib import Path

TIMIT_ROOT = Path("data") / "raw_timit" / "data"
LISTS_DIR = Path("data") / "lists"


def prep_timit_lists():
    LISTS_DIR.mkdir(exist_ok=True, parents=True)

    ubm_list_path = LISTS_DIR / "ubm_train_list.txt"
    enroll_list_path = LISTS_DIR / "test_enrollment_list.txt"

    test_speakers_dict = {}

    print("Looking for UBM data")
    with open(ubm_list_path, 'w') as ubm_file:
        train_dir = TIMIT_ROOT / "TRAIN"
        for wav_path in train_dir.rglob("*.WAV.wav"):
            if not wav_path.name.startswith("SA"):
                ubm_file.write(f"{wav_path.absolute()}\n")
    
    print("Looking thro test dir for speaker data")
    test_dir = TIMIT_ROOT / "TEST"
    for wav_path in test_dir.rglob("*.WAV.wav"):
        if not wav_path.name.startswith("SA"):
            speaker_id = wav_path.parent

            if speaker_id not in test_speakers_dict:
                test_speakers_dict[speaker_id] = []
            
            test_speakers_dict[speaker_id].append(str(wav_path.absolute()))
    
    with open(enroll_list_path, "w") as enroll_file:
        # Format -> SpeakerID|path1,path2,path3... 
        for speakers, paths in test_speakers_dict.items():
            paths_str = ",".join(paths)
            enroll_file.write(f"{speakers}|{paths_str}\n")
    
    print(f"Data lists are generated in {LISTS_DIR.absolute()}")


def main() -> None:
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--timit-root", type=Path, help="Path to timit dataset")
    # ap.add_argument("--lists-root", type=Path, help="Path to the lists directory")
    # args = ap.parse_args()

    # timit_root = args.timit_root
    # lists_root = args.lists_root
    prep_timit_lists()


if __name__ == "__main__":
    main()
