
def color_splash(image, mask):
    """Apply color splash effect.
    :param image: RGB image [height, width, 3]
    :param mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        print("mask.shape[0] !> 0 ".format(mask.shape[0]))
        splash = gray

    return splash


def _visualize(image, rois, masks, class_ids, class_names, scores, ax, title="Predictions", filename="_visualize.png"):
    display_instances(image, rois, masks, class_ids,
                                class_names, scores, ax=ax, title=title, output=filename)


def _get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))

    return ax


def detect_and_color_splash(m, image_path=None, video_path=None):
    global file_name
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = m.detect([image], verbose=1)[0]
        # Color splash
        if len(r['rois']) == 0:
            print("ROIS     : {}".format(r['rois']))
            print("CLASS_IDs: {}".format(r['class_ids']))
            dataset = HomeObjectDataset()
            dataset.load_homeobject(args.dataset, 'Val')
            dataset.prepare()
            ax = _get_ax()

            _visualize(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax,
                       filename=os.path.basename(image_path))
            splash = color_splash(image, r['masks'])
            # Save output
            file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            skimage.io.imsave(file_name, splash)
        else:
            print("No object detected")

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = m.detect([image], verbose=0)[0]
                print("ROIS: {}".format(r['rois']))
                print("CLASS_IDS: {}".format(r['class_ids']))
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
        print("Saved to ", file_name)
