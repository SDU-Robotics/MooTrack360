"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  Computes IOU between two sets of bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / (((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])) + 
            ((bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])) - wh)
  return o  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box, s is the scale/area, and r is
    the aspect ratio.
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w / 2.
  y = bbox[1] + h / 2.
  s = w * h    # scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    # Ensure x is a flat NumPy array.
    x = np.array(x).flatten()
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., float(score)]).reshape((1, 5))



class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  Now it also stores the last detection score and class_id.
  """
  count = 0

  def __init__(self, bbox):
    """
    Initializes a tracker using the initial bounding box.
    bbox is expected to be a numpy array of at least 5 elements:
      [x1, y1, x2, y2, score, (optional) class_id]
    """
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]])
    self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0]])

    self.kf.R[2:, 2:] *= 10.
    self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1, -1] *= 0.01
    self.kf.Q[4:, 4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

    # Store additional info from the detection.
    if bbox.shape[0] >= 6:
      self.last_score = bbox[4]
      self.last_class = bbox[5]
    else:
      self.last_score = 0.0
      self.last_class = -1

  def update(self, bbox):
    """
    Updates the state vector with the observed bbox and updates score and class_id.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))
    if bbox.shape[0] >= 6:
      self.last_score = bbox[4]
      self.last_class = bbox[5]

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if ((self.kf.x[6] + self.kf.x[2]) <= 0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if (self.time_since_update > 0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x, score=self.last_score))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate with score and class_id.
    Output shape: (1, 6) -> [x1, y1, x2, y2, score, class_id]
    """
    bbox_with_score = convert_x_to_bbox(self.kf.x, score=self.last_score)  # shape (1,5)
    # Append class_id as a new column.
    return np.concatenate((bbox_with_score, np.array([[self.last_class]])), axis=1)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
  """
  Assigns detections to tracked objects (both represented as bounding boxes).

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers.
  """
  if len(trackers) == 0:
    return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0, 2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if d not in matched_indices[:, 0]:
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if t not in matched_indices[:, 1]:
      unmatched_trackers.append(t)

  # Filter out matched with low IOU.
  matches = []
  for m in matched_indices:
    if iou_matrix[m[0], m[1]] < iou_threshold:
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1, 2))
  if len(matches) == 0:
    matches = np.empty((0, 2), dtype=int)
  else:
    matches = np.concatenate(matches, axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT.
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 6))):
    """
    Params:
      dets - a numpy array of detections in the format:
             [[x1,y1,x2,y2,score,class_id], ...]
             For frames without detections, pass np.empty((0, 6)).
    Returns an array where the last two columns are the score and class_id,
    and the final appended column is the track ID.
    Output for each track: [x1,y1,x2,y2,score,class_id,track_id]
    NOTE: The number of objects returned may differ from the number of detections.
    """
    self.frame_count += 1
    # Get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]  # pos is [x1,y1,x2,y2,score]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # Update matched trackers with assigned detections.
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # Create and initialise new trackers for unmatched detections.
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i, :])
      self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state()[0]  # d is [x1,y1,x2,y2,score,class_id]
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        # Append track id (+1 so that MOT benchmark IDs are positive).
        ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
      i -= 1
      # Remove dead tracklet.
      if trk.time_since_update > self.max_age:
        self.trackers.pop(i)
    if len(ret) > 0:
      return np.concatenate(ret)
    return np.empty((0, 7))


def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='SORT demo')
  parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]', action='store_true')
  parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
  parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
  parser.add_argument("--max_age", help="Maximum number of frames to keep alive a track without associated detections.", type=int, default=1)
  parser.add_argument("--min_hits", help="Minimum number of associated detections before track is initialised.", type=int, default=3)
  parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  # All train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3)  # used only for display
  if display:
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
      print("Processing %s." % (seq))
      for frame in range(int(seq_dets[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0] == frame, 2:8]
        # If detections are in [x1,y1,w,h,score,class_id] format, convert to [x1,y1,x2,y2,score,class_id]
        dets[:, 2:4] += dets[:, 0:2]
        total_frames += 1

        if display:
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))
          im = io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          # d: [x1,y1,x2,y2,score,class_id,track_id]
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[6], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
          if display:
            d_int = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d_int[0], d_int[1]), d_int[2] - d_int[0], d_int[3] - d_int[1], fill=False, lw=3, ec=colours[d_int[6] % 32, :]))

        if display:
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if display:
    print("Note: to get real runtime results run without the option: --display")
