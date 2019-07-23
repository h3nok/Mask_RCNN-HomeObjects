
from home_objects import AugmentType
from evaluate_model import *
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        default='train',
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=True,
                        default="D:\\HomeObjects06",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        default='coco',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=" D:\\HomeObjects06\\Logs",
                        metavar="Path",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--categories', required=False, type=int,
                        default=56,
                        help="Perform training for sparse or fine grained classification")
    parser.add_argument('--augment', required=False, type=bool,
                        default=True,
                        help="Train with in-training augmentation")
    parser.add_argument('--augment_type', required=False, type=AugmentType,
                        default=AugmentType.Sequential,
                        help="Determines how augmentation is applied during training")
    parser.add_argument('--layers', required=False, type=str,
                        default='all',
                        help='Train all layers or just heads')
    parser.add_argument('--steps', required=False, type=int,
                        default=100,
                        help="Training steps per epochs")
    parser.add_argument('--mrcnn_bbox_loss', required=False, type=str,
                        default=None,
                        help="loss function for mrcnn_bbox_loss")
    parser.add_argument('--mrcnn_class_loss', required=False, type=str,
                        default=None,
                        help="loss function for mrcnn_class_loss")
    parser.add_argument('--rpn_bbox_loss', required=False, type=str,
                        default=None,
                        help="loss function for rpn_bbox_loss")
    parser.add_argument('--rpn_class_loss', required=False, type=str,
                        default=None,
                        help="loss function for rpn_class_loss")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HomeObjectConfig()
        config.NUM_CLASSES = args.categories + 1
        config.STEPS_PER_EPOCH = args.steps
    else:
        class InferenceConfig(HomeObjectConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load_homeobject
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits",
                                                                "mrcnn_bbox_fc", "mrcnn_bbox",
                                                                "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.layers, args.augment, args.augment_type)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image, video_path=args.video)
    else:
        print("'{}' is not recognized. Use 'train' or 'splash'".format(args.command))

