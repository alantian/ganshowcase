#!/usr/bin/env python3

import glob
import sys
import os
import queue
import threading
import traceback

from absl import app
from absl import flags
from absl import logging
import h5py
import numpy as np
from PIL import Image


class ExceptionInfo(object):
    def __init__(self):
        self.type, self.value = sys.exc_info()[:2]
        self.traceback = traceback.format_exc()


class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:  # noqa: E722
                result = ExceptionInfo()
            result_queue.put((result, args))


class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__')  # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func, verbose_exceptions=True):  # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            if verbose_exceptions:
                print('\n\nWorker thread caught an exception:\n' + result.traceback + '\n', end=' ')
            raise result.type(result.value)
        return result, args

    def finish(self):
        for idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self):  # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self,
                                   item_iterator,
                                   process_func=lambda x: x,
                                   pre_func=lambda x: x,
                                   post_func=lambda x: x,
                                   max_items_in_flight=None):
        if max_items_in_flight is None:
            max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, idx):
            return process_func(prepared)

        def retire_result():
            processed, (prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result():
                    yield res
        while retire_idx[0] < len(results):
            for res in retire_result():
                yield res


FLAGS = flags.FLAGS

flags.DEFINE_string('task', '', '')
flags.DEFINE_string('dir_path', '', '')
flags.DEFINE_string('npz_path', '', '')
flags.DEFINE_string('multisize_h5_path', '', '')
flags.DEFINE_integer('size', 64, 'Size for images')
flags.DEFINE_integer('max_images', -1, 'Max number of images to process. -1 for no limitation.')
flags.DEFINE_integer('num_threads', 40, 'Number of concurrent threads.')
flags.DEFINE_integer('num_tasks', 600, 'Number of concurrent processing tasks.')


def multisize_h5_to_npz():
    output_dir = os.path.dirname(FLAGS.npz_path)
    os.system('mkdir -p %s' % output_dir)

    logging.info('Loading from %s for size %d.', FLAGS.npz_path, FLAGS.size)
    h5_file = h5py.File(FLAGS.multisize_h5_path, 'r')
    dset = h5_file['data%dx%d' % (FLAGS.size, FLAGS.size)]

    arr = np.array(dset)
    if FLAGS.max_images > -1:
        arr = arr[:FLAGS.max_images]

    logging.info('%d images laoded', len(arr))
    logging.info('Saving numpy array to %s.', FLAGS.npz_path)
    np.savez(FLAGS.npz_path, **{'size_%s' % (FLAGS.size): arr})


def npz_to_dir():
    output_dir = FLAGS.dir_path
    os.system('mkdir -p %s' % output_dir)

    blob = np.load(FLAGS.npz_path)
    arr = blob['size_%s' % (FLAGS.size)]
    if FLAGS.max_images > -1:
        arr = arr[:FLAGS.max_images]
    logging.info('%d images found', len(arr))

    def process_func(inputs):
        x, save_path = inputs
        x = x.transpose(1, 2, 0)
        x = np.round(x).astype(np.uint8)
        img = Image.fromarray(x, mode='RGB')
        img.save(save_path)
        return None

    imgs = []
    print()
    nb_images = len(arr)
    data_iter = [(x, os.path.join(output_dir, '%08d.png' % id_)) for id_, x in enumerate(arr)]
    for data_item in data_iter:
        imgs.append(process_func(data_item))
        print('%d / %d\r' % (len(imgs), nb_images), end=' ')
    print()
    logging.info('Processed %d images.' % nb_images)


def dir_to_npz():
    output_dir = os.path.dirname(FLAGS.npz_path)
    os.system('mkdir -p %s' % output_dir)

    logging.info('Creating custom dataset %s from %s' % (FLAGS.npz_path, FLAGS.dir_path))
    glob_pattern = os.path.join(FLAGS.dir_path, '*')
    image_filenames = sorted(glob.glob(glob_pattern))
    if len(image_filenames) == 0:
        logging.error('Error: No input images found in %s' % glob_pattern)
        return

    img = np.asarray(Image.open(image_filenames[0]))
    channels = img.shape[2] if img.ndim == 3 else 1

    if channels not in [1, 3]:
        logging.error('Error: Input images must be stored as RGB or grayscale')
        return

    def center_crop(img):
        width, height = img.size  # Get dimensions

        new_width, new_height = min(width, height), min(width, height)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        return img.crop((left, top, right, bottom))

    def process_func(image_filename):
        img = Image.open(image_filename)

        img = center_crop(img)

        img = img.resize((FLAGS.size, FLAGS.size), Image.ANTIALIAS)
        img = np.asarray(img)
        img = img.transpose(2, 0, 1)  # HWC => CHW

        return img

    if FLAGS.max_images > -1:
        image_filenames = image_filenames[:FLAGS.max_images]

    with ThreadPool(FLAGS.num_threads) as pool:
        imgs = []
        print()
        for img in pool.process_items_concurrently(
                image_filenames, process_func=process_func, max_items_in_flight=FLAGS.num_tasks):
            imgs.append(img)
            print('%d / %d\r' % (len(imgs), len(image_filenames)), end=' ')
        print()
        logging.info('Added %d images.' % len(image_filenames))

    logging.info('Converting to numpy array')
    imgs = np.array(imgs, dtype=np.uint8)

    logging.info('Saving numpy array to %s.', FLAGS.npz_path)
    np.savez(FLAGS.npz_path, **{'size_%s' % (FLAGS.size): imgs})


def main(argv):
    del argv  # Unused.

    logging.info('task is %s.' % FLAGS.task)
    func = globals()[FLAGS.task]
    func()


if __name__ == '__main__':
    app.run(main)
