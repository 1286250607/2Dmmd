import os
import sys
import skimage.io
import matplotlib.pyplot as plt


ROOT_DIR = ('./')

sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


def get_ax(rows=1, cols=1, size=20):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax
def main(model_file, img_file):
    MODEL_DIR = os.path.join(ROOT_DIR, "model/")
    model_file = model_file
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, model_file)

    import tdmcoco
    config = tdmcoco.CocoConfig()
    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 3
        DETECTION_MIN_CONFIDENCE = 0.7
    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    model.load_weights(COCO_MODEL_PATH, by_name=True)

    ax = get_ax(1)
    class_names = ['', 'Mono', 'Few', 'Thick']
    fig, ax = plt.subplots()

    IMAGE_DIR = os.path.join(ROOT_DIR, "img/input/")
    img_file = img_file
    USE_IMAGE_DIR = os.path.join(IMAGE_DIR, img_file)
    image = skimage.io.imread(USE_IMAGE_DIR)

    results = model.detect([image], verbose=1)

    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax=ax)

    ax.axis('off')
    plt.tight_layout()

    M = int(0)
    F = int(0)
    T = int(0)
    d = {}
    for i in r['class_ids']:
        if i == 1:
            M += 1
        elif i== 2:
            F += 1
        elif i == 3:
            T += 1
    d['Mono'] = M
    d['Few'] = F
    d['Thick'] = T
    if M == 0 and F == 0 and T == 0:
        return False
    else:
        plt.savefig(f'./img/output/{img_file}', bbox_inches='tight')
        return d




