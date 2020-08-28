import os
import cv2
import time
import tensorflow as tf
import numpy as np

from fly4tf.utils import label_map_util
from fly4tf.utils import image_processing
from fly4tf.utils import visualization_utils2 as vis_util2

BASE_PATH = os.path.dirname(__file__)
PATH_TO_IMAGE_PROCESSING = os.path.join(BASE_PATH, '../image_processing_folder')
PATH_TO_GRAPH = os.path.join(BASE_PATH, 'inference_graph', 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(BASE_PATH,'inference_graph','labelmap.pbtxt')
NUM_CLASSES = 1
DRAW_BBOX = True


def _category_index() -> dict:
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    return label_map_util.create_category_index(categories)


def _session(detection_graph: tf.Graph) -> tf.compat.v1.Session:
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return tf.compat.v1.Session(graph=detection_graph, config=tf.compat.v1.ConfigProto())


def _tensors(detection_graph: tf.Graph) -> tuple:
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return image_tensor, detection_boxes, detection_scores, detection_classes, num_detections


class ResultOfCounting:
    def __init__(self, img, x, y, count) -> None:
        self.img = img
        self.x = x
        self.y = y
        self.count = count


def _draw_result_img(filename, img, count):
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    t, fontScale = 4, 2.0
    cv2.putText(img, str(count), (50, 50), fontFace, fontScale, (36, 255, 12), t)
    cv2.imwrite(os.path.join(PATH_TO_IMAGE_PROCESSING, filename), img)


def main() -> None:
    category_index = _category_index()
    # Load the TensorFlow model into memory.
    detection_graph = tf.Graph()
    session = _session(detection_graph)
    # Define input and output tensors (i.e. data) for the object detection classifier
    image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = _tensors(detection_graph)

    def _object_counting(img_part) -> ResultOfCounting:
        start_time = time.time()
        _img, _x, _y = img_part
        print("Start {0} for {1}x{2}".format(start_time, _x, _y))
        image_expanded = np.expand_dims(_img, axis=0)
        (boxes, scores, classes, num) = session.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        if DRAW_BBOX:
            counter, csv_line, counting_mode = vis_util2.visualize_boxes_and_labels_on_image_array(
                0, _img, 1, 0, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                category_index, max_boxes_to_draw=40, use_normalized_coordinates=True, line_thickness=4)
            count = 0
            str_count = counting_mode.replace("'larve:': ", '')
            if str_count:
                count = int(str_count)
        else:
            count = scores[np.where(scores > .5)].shape[0]
        print("Finish {0} for {1}x{2}".format(time.time() - start_time, _x, _y))
        return ResultOfCounting(_img, _x, _y, count)

    img_files = [f for f in os.listdir(PATH_TO_IMAGE_PROCESSING) if
                 os.path.isfile(os.path.join(PATH_TO_IMAGE_PROCESSING, f))
                 and not f.startswith('res_') and not f == '.dummy']
    for img_file in img_files:
        start_processing_time = time.time()
        # Processing image
        image = cv2.imread(os.path.join(PATH_TO_IMAGE_PROCESSING, img_file))
        h, w = image.shape[:2]
        contours = image_processing.findContours(image)
        blank = np.zeros((h, w, 1), dtype=np.uint8)
        mask = cv2.fillPoly(blank, contours, (255, 255, 255))
        image = cv2.bitwise_and(image, image, mask=mask)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _h, _w = 432, 648
        img_parts = image_processing.sliceImage(image_rgb, _h, _w)
        total_count = 0
        result_img = np.zeros((h, w, 3), dtype=np.uint8)
        for img_p in img_parts:
            res = _object_counting(img_p)
            total_count += res.count
            result_img[res.y:res.y + _h, res.x:res.x + _w] = cv2.cvtColor(res.img, cv2.COLOR_RGB2BGR)
        # with ThreadPoolExecutor(2) as executor:
        #     total_count = 0
        #     result_img = np.zeros((h, w, 3), dtype=np.uint8)
        #     for res in executor.map(_object_counting, img_parts):
        #         total_count += res.count
        #         result_img[res.y:res.y + _h, res.x:res.x + _w] = cv2.cvtColor(res.img, cv2.COLOR_RGB2BGR)
        print("Count: {0}. Seconds {1}".format(total_count, time.time() - start_processing_time))
        print('-------------------------------------')
        _draw_result_img('res_' + img_file, result_img, total_count)
    session.close()


if __name__ == '__main__':
    main()
