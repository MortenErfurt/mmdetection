# The new config inherits a base config to highlight the necessary modification
_base_ = '../cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(num_classes=1),
            dict(num_classes=1),
            dict(num_classes=1)
        ]))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('person')
data = dict(
    train=dict(
        img_prefix='configs/cctv/train/',
        classes=classes,
        ann_file='configs/cctv/train/annotation_coco.json'),
    val=dict(
        img_prefix='configs/cctv/val/',
        classes=classes,
        ann_file='configs/cctv/val/annotation_coco.json'),
    test=dict(
        img_prefix='configs/cctv/val/',
        classes=classes,
        ann_file='configs/cctv/val/annotation_coco.json'))

# We can use the pre-trained Cascade RCNN model to obtain higher performance
load_from = 'pretrained_models/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'