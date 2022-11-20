# Example:
# python3 extract_images_from_tensorboard.py events_root_path/ "Image/Render (val)"

import tempfile
import subprocess
import cv2
import numpy as np
import os
from pathlib import Path
import imageio
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def run_subprocess(command, working_dir=None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=working_dir)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)

# Source: https://stackoverflow.com/questions/47232779/how-to-extract-and-save-images-from-tensorboard-event-summary
def get_images_from_event(fn, tag):
    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    yield np.asarray(im)

if __name__ == '__main__':
    import sys
    event_root_path, tag_name = sys.argv[1:]
    event_root_path = Path(event_root_path)

    video_writer = None
    video_writer_output_path = event_root_path / (tag_name.replace("/", "_") + "_tmp.mp4")
    final_video_output_path = event_root_path / (tag_name.replace("/", "_") + ".mp4")

    event_files = sorted(
        x for x in event_root_path.iterdir() if x.name.startswith("events.out.tfevents."))

    if not event_files:
        raise FileNotFoundError(event_root_path)
    else:
        print(event_files)

    for event_file in event_files:
        for image in get_images_from_event(str(event_file), tag_name):
            if video_writer is None:
                h, w, _ = image.shape
                video_writer = cv2.VideoWriter(
                    str(video_writer_output_path),
                    cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))

            video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    video_writer.release()

    run_subprocess(["ffmpeg", "-i", str(video_writer_output_path), "-c:v", "h264", "-crf", "13", str(final_video_output_path)])
    video_writer_output_path.unlink()
