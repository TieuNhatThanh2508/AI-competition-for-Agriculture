'''
Inpainting and Argument Recongnition configure file
'''
img_size=256
mode='train'
config=dict(
    name='VRNetsWithInpaintingAGRExamplarFT',
    backbone_cfg=dict(
        name='resnet50',
        num_classes=None,
        in_channels=36,
        pretrained=False,
        out_keys=('block2','block3','block4','block5')
    ),
    inpainting_head_cfg=dict(
        name='InpaintingHead',
        in_channels=2048,
        out_channels=36,
        img_size=img_size,
        feat_channels=(1024,512,256,64,32)
    ),
    agr_head_cfg=dict(
        name='AGPHead',
        in_channels=2048,
        num_classes=6,
    ),
    examplar_head_cfg=dict(
        name='ExamplarHead',
        in_channels=2048,
        out_channels=512
    ),
    train_cfg=dict(
        batch_size=4,
        device='cuda:0',
        num_epoch=4,
        num_workers=4,
        train_data=dict(

            data_path='D:/s2a demo/',
            # data_format='*.tif',
            img_size=256,
            inpainting_transforms_cfg=dict(
                min_cover_ratio=0.2,max_cover_ratio=1./3,
                 brightness=0.3,contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.3,0.3)
            ),
            pre_transforms_cfg=dict(
                RandomHorizontalFlip=dict(name='RandomHorizontalFlip'),
                RandomVerticalFlip=dict(name='RandomVerticalFlip'),
                Rotate=dict(name='Rotate'),
            ),

            agr_transforms_cfg=dict(
                transforms_cfg=dict(
                    RandomHorizontalFlip=dict(name='RandomHorizontalFlip',p=1.),
                    RandomVerticalFlip=dict(name='RandomVerticalFlip',p=1.),
                    Rotate_0=dict(name='Rotate',angle=0,p=1),
                    Rotate_90=dict(name='Rotate',angle=90,p=1),
                    Rotate_180=dict(name='Rotate',angle=180,p=1),
                    Rotate_270=dict(name='Rotate',angle=270,p=1),
                    # RandomCrop=dict(name='RandomCrop', crop_ratio_min=0.7, crop_ratio_max=0.95),
                    # ColorJitter=dict(name='ColorJitter', brightness=0.3, contrast=(0.5, 1.5), saturation=(0.5, 1.5),
                    #                  hue=(-0.3, 0.3)),
                ),
                shortcut_cfg=dict(
                    RandomCrop=dict(name='RandomCrop', crop_ratio_min=0.7, crop_ratio_max=0.95),
                    ColorJitter=dict(name='ColorJitter', brightness=0.3, contrast=(0.5, 1.5), saturation=(0.5, 1.5),
                                     hue=(-0.3, 0.3)),
                    RandomGrayscale=dict(name='RandomGrayscale',p=0.5),
                    # Normal=dict(name='Normal')
                ),
            ),
            post_transforms_cfg=dict(
                Resize=dict(name='Resize',size=(256,256)),
                ToTensor=dict(name='ToTensor')
            ),
        ),

        losses=dict(
            InpaintingLoss=dict(name='InpaintingLoss'),
            AGRLoss=dict(name='CrossEntropyLoss'),
            ExamplarLoss=dict(name='ExamplarLoss'),
            factors=[20,1,1]
        ),

        optimizer=dict(
            name='Adam',
            lr=0.0005
        ),

        checkpoints=dict(
            checkpoints_path=r'checkpoints/checkpoints_resnet50_imagenet_inpainting_agr_examplar_total',
            save_step=1,
        ),
        lr_schedule=dict(
            name='stepLR',
            step_size=2,
            gamma=0.95
        ),
        log=dict(
            log_path=r'log/log_resnet50_inpainting_agr_examplar_total',
            log_step=50,
            with_vis=False,
            vis_path=r''
        ),
    ),
    test_cfg=dict(
        batch_size=1,
        test_data=dict(
            data_path='E:/SSLRemoteSensing/SSLRemoteSensing/datasets/datasets/test_data/*/',
            label_path='E:/SSLRemoteSensing/SSLRemoteSensing/datasets/datasets/test_labels/*/',
            data_format='*.tif',
            label_format='*.tif'
        ),
        checkpoints=dict(
            checkpoints_path=r'checkpoints/checkpoints_resnet50_imagenet_inpainting_agr_examplar_total',
        ),
        log=dict(
            with_vis=True,
            vis_path=r'log/vis_resnet50_inpainting_agr_examplar_total_test'
        ),
    ),
    fine_tune_cfg=dict(
        batch_size=4,
        device='cuda:0',
        num_epoch=20,
        num_workers=4,
        train_data=dict(

            data_path='C:/Users/ADMIN/OneDrive/Desktop/VuonAI/AI FOR AGRICULTURE/Dataset/ICPR02/data demo/',
            # data_format='*.tif',
            img_size=256,
            inpainting_transforms_cfg=dict(
                min_cover_ratio=0.2,max_cover_ratio=1./3,
                 brightness=0.3,contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.3,0.3)
            ),
            pre_transforms_cfg=dict(
                RandomHorizontalFlip=dict(name='RandomHorizontalFlip'),
                RandomVerticalFlip=dict(name='RandomVerticalFlip'),
                Rotate=dict(name='Rotate'),
            ),

            agr_transforms_cfg=dict(
                transforms_cfg=dict(
                    RandomHorizontalFlip=dict(name='RandomHorizontalFlip',p=1.),
                    RandomVerticalFlip=dict(name='RandomVerticalFlip',p=1.),
                    Rotate_0=dict(name='Rotate',angle=0,p=1),
                    Rotate_90=dict(name='Rotate',angle=90,p=1),
                    Rotate_180=dict(name='Rotate',angle=180,p=1),
                    Rotate_270=dict(name='Rotate',angle=270,p=1),
                    # RandomCrop=dict(name='RandomCrop', crop_ratio_min=0.7, crop_ratio_max=0.95),
                    # ColorJitter=dict(name='ColorJitter', brightness=0.3, contrast=(0.5, 1.5), saturation=(0.5, 1.5),
                    #                  hue=(-0.3, post_transforms0.3)),
                ),
                shortcut_cfg=dict(
                    RandomCrop=dict(name='RandomCrop', crop_ratio_min=0.7, crop_ratio_max=0.95),
                    ColorJitter=dict(name='ColorJitter', brightness=0.3, contrast=(0.5, 1.5), saturation=(0.5, 1.5),
                                     hue=(-0.3, 0.3)),
                    RandomGrayscale=dict(name='RandomGrayscale',p=0.5),
                    # Normal=dict(name='Normal')
                ),
            ),
            post_transforms_cfg=dict(
                Resize=dict(name='Resize',size=(256,256)),
                ToTensor=dict(name='ToTensor')
            ),
        ),

        losses=dict(
            InpaintingLoss=dict(name='InpaintingLoss'),
            AGRLoss=dict(name='CrossEntropyLoss'),
            ExamplarLoss=dict(name='ExamplarLoss'),
            factors=[10,1,1,10]
        ),

        optimizer=dict(
            name='Adam',
            lr=0.0005
        ),

        checkpoints=dict(
            checkpoints_path=r'checkpoints/checkpoints_resnet50_imagenet_inpainting_agr_examplar_total_finetune',
            save_step=1,
        ),
        lr_schedule=dict(
            name='stepLR',
            step_size=2,
            gamma=0.95
        ),
        log=dict(
            log_path=r'log/log_resnet50_inpainting_agr_examplar_total',
            log_step=50,
            with_vis=False,
            vis_path=r''
        ),
    ),
)
