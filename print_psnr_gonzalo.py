# Usage: python3 print_psnr_gonzalo.py 100_rank400_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr2.5e4_camFt

import pathlib
import sys
import tqdm

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_PSNR(event_file_path, pick_criterion='last', verify_step=None):
    """
    event_file_path
        pathlib.Path or str
        Path to a Tensorboard event file.
    pick_criterion
        str
        'last' (return last validation step's PSNR) or 'max' (return max PSNR among all steps).
    verify_step
        int or None
        If not None, test if the picked PSNR has this step number. If not, fail.
    """
    event_file_content = EventAccumulator(str(event_file_path))
    event_file_content.Reload()

    if pick_criterion == 'last':
        pick_criterion_fn = lambda x: x.step
    elif pick_criterion == 'max':
        pick_criterion_fn = lambda x: x.value
    else:
        raise ValueError(f"Wrong pick_criterion: {pick_criterion}")
    last_PSNR = max(event_file_content.Scalars('Loss/PSNR (val)'), key=pick_criterion_fn)
    if verify_step:
        assert last_PSNR.step == verify_step, f"Expected step {verify_step} but found {last_PSNR.step} in {event_file_path}"
    return last_PSNR.value

run_name = sys.argv[1]

for iteration in '18000', '38000':
    psnr = []

    for view_name in tqdm.tqdm(('left-1', 'frontal-1', 'right-1', 'LR-2', 'LFR-3')):
        for scene_name in ('130', '132', '134', '149'):
            run_root_dir = pathlib.Path(f"logs-paper/gonzalo_val-cameras-opt/{run_name}/{view_name}/{scene_name}/{iteration}")

            event_files = sorted(x for x in run_root_dir.iterdir() if x.name.startswith("events.out.tfevents"))
            assert len(event_files) == 1

            psnr.append(get_PSNR(event_files[0], pick_criterion='max'))

    print(' '.join(map(str, psnr)))
