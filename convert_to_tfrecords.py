import os
import numpy as np
#import h5py
#import random
from PIL import Image
#import tensorflow as tf
#from meta import Meta
 
#tf.app.flags.DEFINE_string('data_dir', './data',
#                           'Directory to SVHN (format 1) folders and write the converted files')
#FLAGS = tf.app.flags.FLAGS
FLAGS_data_dir = 'data'
 
class ExampleReader(object):
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._example_pointer = 0
 
    @staticmethod
    def _get_attrs(digit_csv_file, index):
        """
        Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
        """
#        attrs = {}
#        f = digit_csv_file
#        item = f['digitStruct']['bbox'][index].item()
#        for key in ['label', 'left', 'top', 'width', 'height']:
#            attr = f[item][key]
#            values = [f[attr.value[i].item()].value[0][0]
#                      for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
#            attrs[key] = values
        attrs = {
            'label' : digit_csv_file[index][0]
        }
        return attrs
 
    @staticmethod
    def _preprocess(image, bbox_left, bbox_top, bbox_width, bbox_height):
#        cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
#                                                                    int(round(bbox_top - 0.15 * bbox_height)),
#                                                                    int(round(bbox_width * 1.3)),
#                                                                    int(round(bbox_height * 1.3)))
#        image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
#        image = image.resize([64, 64])
#        image.show()
        image = image.resize([128, 32])
#        image.show()
        return image
 
#    @staticmethod
#    def _int64_feature(value):
#        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#    @staticmethod
#    def _float_feature(value):
#        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#
#    @staticmethod
#    def _bytes_feature(value):
#        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
    @staticmethod
    def _char_to_digit(c):
        if c == '.': return 10
        if c == ':': return 11
        return(int(c))
       
    def read_and_convert(self, digit_csv_file):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]
        index = int(path_to_image_file.split('\\')[-1].split('.')[0]) - 1
        self._example_pointer += 1
 
        attrs = ExampleReader._get_attrs(digit_csv_file, index)
        digit_label = attrs['label']
        print("digit_label: %s" % digit_label)
        length = len(digit_label)
        if length > 8:
            # Skip this example
            print("Skipped example as number of digits for [%s] is %d" % (path_to_image_file, length))
            return
 
        digits = [12, 12, 12, 12, 12, 12, 12, 12]   # 12 represents no digit
        for idx in range(length):
            digits[8 - length + idx] = self._char_to_digit(digit_label[idx])
        print("digits: %s" % digits)
#        attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
#        min_left, min_top, max_right, max_bottom = (min(attrs_left),
#                                                    min(attrs_top),
#                                                    max(map(lambda x, y: x + y, attrs_left, attrs_width)),
#                                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))
        raw_image = Image.open(path_to_image_file)
        (width, height) = raw_image.size
        min_left, min_top, max_right, max_bottom = (1,
                                                    1,
                                                    width,
                                                    height)                                                   
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                        center_y - max_side / 2.0,
                                                        max_side,
                                                        max_side)
        image = np.array(ExampleReader._preprocess(raw_image, bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()
 
#        example = tf.train.Example(features=tf.train.Features(feature={
#            'image': ExampleReader._bytes_feature(image),
#            'length': ExampleReader._int64_feature(length),
#            'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
#        }))
        example = {
            'image'  : image,
            'length' : length,
            'digits' : digits
        }
        print("final image size (%dx%d): %d" % (width, height, len(example['image'])))
        return example
 
def convert_to_tfrecords(path_to_dataset_dir_and_digit_csv_file_tuples,
                         path_to_tfrecords_files, choose_writer_callback):
    num_examples = []
    writers = []
 
    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
#        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))
 
    for path_to_dataset_dir, path_to_digit_csv_file in path_to_dataset_dir_and_digit_csv_file_tuples:
#        path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.png'))
        path_to_image_files = [os.path.join(path_to_dataset_dir, '1.png'),
                               os.path.join(path_to_dataset_dir, '2.png'),
                               os.path.join(path_to_dataset_dir, '3.png'),
                               os.path.join(path_to_dataset_dir, '4.png'),
                               os.path.join(path_to_dataset_dir, '5.png'),
                               os.path.join(path_to_dataset_dir, '6.png')]
        total_files = len(path_to_image_files)
        print '%d files found in %s' % (total_files, path_to_dataset_dir)
 
#        with h5py.File(path_to_digit_csv_file, 'r') as digit_csv_file:
#            example_reader = ExampleReader(path_to_image_files)
#            for index, path_to_image_file in enumerate(path_to_image_files):
#                print '(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file)
#
#                example = example_reader.read_and_convert(digit_csv_file)
#                if example is None:
#                    break
#
#                idx = choose_writer_callback(path_to_tfrecords_files)
#                writers[idx].write(example.SerializeToString())
#                num_examples[idx] += 1
        digit_csv_file = np.genfromtxt(path_to_digit_csv_file,
                                       dtype = ['S8'],
                                       delimiter = ',')
        example_reader = ExampleReader(path_to_image_files)
        for index, path_to_image_file in enumerate(path_to_image_files):
            print '(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file)
 
            example = example_reader.read_and_convert(digit_csv_file)
            if example is None:
                break
 
#            idx = choose_writer_callback(path_to_tfrecords_files)
#            writers[idx].write(example.SerializeToString())
            idx = 0
            num_examples[idx] += 1
               
    for writer in writers:
        writer.close()
 
    return num_examples
 
def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file):
    print 'Saving meta file to %s...' % path_to_tfrecords_meta_file
#    meta = Meta()
#    meta.num_train_examples = num_train_examples
#    meta.num_val_examples = num_val_examples
#    meta.num_test_examples = num_test_examples
#    meta.save(path_to_tfrecords_meta_file)
 
def main(_):
#    path_to_train_dir = os.path.join(FLAGS.data_dir, 'train')
#    path_to_test_dir = os.path.join(FLAGS.data_dir, 'test')
#    path_to_extra_dir = os.path.join(FLAGS.data_dir, 'extra')
#    path_to_train_digit_csv_file = os.path.join(path_to_train_dir, 'digitStruct.mat')
#    path_to_test_digit_csv_file = os.path.join(path_to_test_dir, 'digitStruct.mat')
#    path_to_extra_digit_csv_file = os.path.join(path_to_extra_dir, 'digitStruct.mat')
#
#    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
#    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
#    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
#    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
   
#    path_to_train_dir = os.path.join(FLAGS_data_dir, 'train')
    path_to_test_dir = os.path.join(FLAGS_data_dir, 'test')
#    path_to_extra_dir = os.path.join(FLAGS_data_dir, 'extra')
#    path_to_train_digit_csv_file = os.path.join(path_to_train_dir, 'digitStruct.mat')
    path_to_test_digit_csv_file = os.path.join(path_to_test_dir, 'digitStruct.csv')
#    path_to_extra_digit_csv_file = os.path.join(path_to_extra_dir, 'digitStruct.mat')
 
#    path_to_train_tfrecords_file = os.path.join(FLAGS_data_dir, 'train.tfrecords')
#    path_to_val_tfrecords_file = os.path.join(FLAGS_data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS_data_dir, 'test.tfrecords')
#    path_to_tfrecords_meta_file = os.path.join(FLAGS_data_dir, 'meta.json')
   
#    for path_to_file in [path_to_train_tfrecords_file, path_to_val_tfrecords_file, path_to_test_tfrecords_file]:
#        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file
 
#    print 'Processing training and validation data...'
#    [num_train_examples, num_val_examples] = convert_to_tfrecords([(path_to_train_dir, path_to_train_digit_csv_file),
#                                                                   (path_to_extra_dir, path_to_extra_digit_csv_file)],
#                                                                  [path_to_train_tfrecords_file, path_to_val_tfrecords_file],
#                                                                  lambda paths: 0 if random.random() > 0.1 else 1)
    print 'Processing test data...'
    [num_test_examples] = convert_to_tfrecords([(path_to_test_dir, path_to_test_digit_csv_file)],
                                               [path_to_test_tfrecords_file],
                                               lambda paths: 0)
 
#    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
#                               path_to_tfrecords_meta_file)
 
    print 'Done %s' % num_test_examples
 
if __name__ == '__main__':
#    tf.app.run(main = main)
    main(None)