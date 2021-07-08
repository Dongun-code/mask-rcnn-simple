"""
just exercise code

"""


from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

path = '/media/milab/My Passport3/Dongun/Segmentation_data/COCO/annotations/instances_train2014.json'

coco = COCO(path)

print(coco.getCatIds())
cats = coco.loadCats(coco.getCatIds())
# print(cats)

names = [cat['name'] for cat in cats]
# print('COCO supercategories: \n{}' .format(' '.join(names)))

catIds = coco.getCatIds(catNms=['person', 'car','bus','bicyle', 'motorcycle'])
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)
imgIds = coco.getImgIds(imgIds= imgIds[0])
img = coco.loadImgs(imgIds)
print(img)
I = io.imread(img[0]['coco_url']) 
# print(I.shape)
plt.axis('off') 
plt.imshow(I) 
annIds = coco.getAnnIds(imgIds=img[0]['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()


