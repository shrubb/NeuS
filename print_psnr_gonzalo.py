# Usage: python3 print_psnr_gonzalo.py 100_rank400_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr2.5e4_camFt

import pathlib
import sys
import tqdm

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_last_PSNR(event_file_path, verify_step=None):
    event_file_content = EventAccumulator(str(event_file_path))
    event_file_content.Reload()
    last_PSNR = max(event_file_content.Scalars('Loss/PSNR (val)'), key=lambda x: x.step)
    if verify_step:
        assert last_PSNR.step == verify_step, f"Expected step {verify_step} but found {last_PSNR.step} in {event_file_path}"
    return last_PSNR.value

run_name = sys.argv[1]

psnr_before_unfreeze = []
psnr_after_unfreeze = []

for view_name in tqdm.tqdm(('left-1', 'frontal-1', 'right-1', 'LR-2', 'LFR-3')):
    for scene_name in ('130', '132', '134', '149'):
        run_root_dir = pathlib.Path(f"logs-paper/gonzalo/{run_name}/{view_name}/{scene_name}")

        event_files = sorted(x for x in run_root_dir.iterdir() if x.name.startswith("events.out.tfevents"))
        assert len(event_files) == 2

        psnr_before_unfreeze.append(get_last_PSNR(event_files[0], verify_step=18000))
        psnr_after_unfreeze.append(get_last_PSNR(event_files[1], verify_step=38000))

print(' '.join(map(str, psnr_before_unfreeze)))
print(' '.join(map(str, psnr_after_unfreeze)))

