# coding=utf-8
from mxnet.gluon.data import Dataset, DataLoader
from utils import generate_rbox, crop_area, get_files, load_annoataion, check_and_validate_polys
import cv2
import numpy as np
from mxnet.gluon.data.vision import transforms
from mxnet import nd
class text_detection_data(Dataset):

    def __init__(self, image_dir, input_size=512, max_large_side=1280, max_text_size=800, min_text_size=3, min_crop_ratio=0.1, \
                 random_scales=np.array([0.5, 1, 1.5, 2.0, 2.5, 3.0]), background_ratio=3./8):
        self.image_dir = image_dir
        self.max_large_side = max_large_side
        self.max_text_size = max_text_size
        self.min_text_size = min_text_size
        self.min_crop_ratio = min_crop_ratio
        self.image_list = get_files(image_dir)
        self.scales = random_scales
        self.bg_ratio = background_ratio
        self.input_size = input_size
        self.seed = np.random.seed(2008)
        self.transform = transforms.Compose([
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])


    def __getitem__(self, item):
        im_fn = self.image_list[item]
        im = cv2.imread(im_fn)

        h, w, _=im.shape
        tmp_fn = im_fn.strip('\n').split('.')
        tmp_fn[-1] = 'txt'
        txt_fn = ".".join(tmp_fn)
        text_polys, text_tags = load_annoataion(txt_fn)
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

        rd_scale = np.random.choice(self.scales)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale
        if np.random.rand() < self.bg_ratio:
            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, self.input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize=(self.input_size, self.input_size))
            score_map = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
            geo_map = np.zeros((self.input_size, self.input_size, 5), dtype=np.uint8)
            training_mask = np.ones((self.input_size, self.input_size), dtype=np.uint8)
        else:
            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
            h, w, _ = im.shape
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, self.input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im.shape
            resize_h = self.input_size
            resize_w = self.input_size
            im = cv2.resize(im, dsize=(resize_w, resize_h))
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)
            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
            new_h, new_w, _ = im.shape
            score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags, self.min_text_size)
        im = self.transform(nd.array(np.transpose(im, (2, 0, 1)).astype('float32')))
        return im, score_map.astype('float32')[np.newaxis, ::4, ::4], geo_map.astype('float32').transpose((2, 0, 1))[:, ::4, ::4], \
               training_mask.astype('float32')[np.newaxis, ::4, ::4]

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    path = '../../../data/icpr/icpr_test'
    td_dataset = text_detection_data(image_dir=path)
    loader = DataLoader(dataset=td_dataset, batch_size=10, num_workers=24)
    print len(loader)
    t1 = time.time()
    for l in loader:
        im, score_map, geo_map, training_mask = l
        print im.shape, score_map.shape, geo_map.shape, training_mask.shape
        im, score_map, geo_map, training_mask = map(lambda x:x.asnumpy(), [im, score_map, geo_map, training_mask])
        fig, axs = plt.subplots(3, 2, figsize=(20, 30))
        # axs[1].set_yticks([])
        axs[0, 0].set_title('im')
        # axs[0, 0].imshow(im[:, :, ::-1].astype(np.uint8))
        axs[0, 0].imshow(im[0][0, :, :].astype(np.uint8))
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].set_title('score_map')
        axs[0, 1].imshow(score_map[0][0, ::, ::])
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[1, 0].set_title('geo_map_0')
        axs[1, 0].imshow(geo_map[0][0, ::, ::])
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(geo_map[0][0, ::, ::])
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])

        plt.tight_layout()
        plt.show()
        plt.close()
        break
    t2 = time.time() - t1
    print 1.0 * len(loader)/t2