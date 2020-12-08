from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import random
import sys
import build_data
from six.moves import range
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

cwd = os.getcwd()

tf.app.flags.DEFINE_string('image_folder',
                     './PQR/JPEGImages',
                     'Folder containing images.')

tf.app.flags.DEFINE_string(
'semantic_segmentation_folder',
'./PQR/SegmentationClassRaw',
'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
'list_folder',
'./PQR/ImageSets',
'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
'output_dir',
'./PQR/tfrecord',
'Path to save converted SSTable of TensorFlow examples.')

_NUM_SHARDS = 4


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        sys.stdout.write(FLAGS.image_folder)
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        image_data = tf.gfile.GFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


# def main(unused_argv):
#   tf.gfile.MakeDirs(FLAGS.output_dir)
#   _convert_dataset(
#       'train', FLAGS.train_image_folder, FLAGS.train_image_label_folder)
#   _convert_dataset('val', FLAGS.val_image_folder, FLAGS.val_image_label_folder)

def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()