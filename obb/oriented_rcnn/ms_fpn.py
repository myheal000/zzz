_base_ = './faster_rcnn_orpn_r50_fpn_3x_hrsc.py'
model = dict(
    neck=
        dict(type='MSFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))