from pathlib import Path
import argparse
import mxnet as mx
from tqdm import tqdm
from PIL import Image
import cv2
import numbers

# Reference: https://github.com/mk-minchul/cse803_face_augmentation/blob/master/preprocessing/training_data/prepare_training_data.py

def save_rec_to_img_dir(rec_path, save_path):
    # open mxrecord file and save training data into rbg images.

    if not save_path.exists():
        save_path.mkdir()

    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        if not isinstance(header.label, numbers.Number):
            label = int(header.label[0])
        else:
            label = int(header.label)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()

        img_save_path = label_path/'{}.jpg'.format(idx)
        img.save(img_save_path, quality=95)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("--rec_path", default='/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/setup_casia_webface/faces_webface_112x112', type=str)
    parser.add_argument("--save_path", default='/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/casia_webface/casia_webface', type=str)

    args = parser.parse_args()
    rec_path = Path(args.rec_path)
    save_path = Path(args.save_path)
    save_rec_to_img_dir(rec_path, save_path)
