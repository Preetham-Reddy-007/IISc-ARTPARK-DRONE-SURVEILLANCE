import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

class VideoRecord(object):
    """
    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])


    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive
    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def label(self):
        # just one label_id
        if len(self._data) == 4:
            return int(self._data[3])
        # sample associated with multiple labels
        else:
            return [int(label_id) for label_id in self._data[3:]]

class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        random_shift: Whether the frames from each segment should be taken
                      consecutively starting from the center of the segment, or
                      consecutively starting from a random location inside the
                      segment range.
        test_mode: Whether this is a test dataset. If so, chooses
                   frames from segments with random_shift=False.

    """
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str='img_{:05d}.jpg',
                 transform = None,
                 random_shift: bool = True,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()
        self._sanity_check_samples()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f"\nDataset Warning: data sample {record.path} seems to have zero RGB frames on disk!\n")

    def _sample_indices(self, record):

        segment_duration = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
        if segment_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), segment_duration) + np.random.randint(segment_duration, size=self.num_segments)

        # edge cases for when a video has approximately less than (num_frames*frames_per_segment) frames.
        # random sampling in that case, which will lead to repeated frames.
        else:
            offsets = np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return offsets

    def _get_val_indices(self, record):

        if record.num_frames > self.num_segments + self.frames_per_segment - 1:
            offsets = self._get_test_indices(record)

        # edge case for when a video does not have enough frames
        else:
            offsets = np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return offsets

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):

        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self._get(record, segment_indices)

    def _get(self, record, indices):

        indices = indices + record.start_frame
        images = list()
        image_indices = list()
        for seg_ind in indices:
            frame_index = int(seg_ind)
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(record.path, frame_index)
                images.extend(seg_img)
                image_indices.append(frame_index)
                if frame_index < record.end_frame:
                    frame_index += 1
        if self.transform is not None:
            images = self.transform(images)

        images = torch.stack([pic for pic in images]).permute(1, 0, 2, 3)
        return images, record.label

    def __len__(self):
        return len(self.video_list)
