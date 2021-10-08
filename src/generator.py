import numpy as np

from tensorflow.keras.utils import to_categorical


def random_location(image_shape, width):

    x = np.random.choice(range(0, image_shape[0] - width))

    y = np.random.choice(range(0, image_shape[1] - width))

    return x, y


def insert_random_box(batch_Y, batch_index, image_shape, box_width, box_class=1):
    
    x, y = random_location(image_shape, box_width)
    batch_Y[batch_index, x:(x+box_width), y:(y+box_width)] = box_class

    return batch_Y


def get_generator(image_shape, n_classes, batch_size=20, box_width_small=16, box_width_large=32):

    classes = range(n_classes)

    while True:

        batch_X = np.empty((batch_size,) + image_shape)

        # Default (background) is class zero
        batch_Y = np.zeros(
            (batch_size,) + image_shape[:2] + (1,), dtype=int
        )

        for batch_index in range(batch_size):

            # Pick a random location to insert a box of class 1
            # TODO Could add a little randomness to the box width to make things more interesting!
            batch_Y = insert_random_box(batch_Y, batch_index, image_shape, box_width_small, box_class=1)

            # Pick a second random location to insert a box of class 2
            # (note that this could cover parts of the first box)
            batch_Y = insert_random_box(batch_Y, batch_index, image_shape, box_width_small, box_class=2)

            # Pick a third random location to insert a large box of class 2
            # (note that this could cover parts of the first two boxes)
            batch_Y = insert_random_box(batch_Y, batch_index, image_shape, box_width_large, box_class=2)

            triangle_noise = np.random.uniform(size=batch_X[batch_index].shape) + np.random.uniform(size=batch_X[batch_index].shape)

            # Note: this means the range of X (when it is not missing) is [0, 4]
            # Lower values are associated with class 0 (background), higher values with classes 1 and 2
            batch_X[batch_index] = batch_Y[batch_index] + triangle_noise

            # Now let's add some random horizontal and vertical "bars" of missing Xs,
            # which we'll encode as -1 (note that this _does not_ overlap with the support for non-missing X)
            # Will the model be able to learn to ignore the missing data, and look at nearby non-missing pixels
            # to figure out the class label?
            x, y = random_location(image_shape, width=2)
            batch_X[batch_index, x:(x+2), :] = -1
            batch_X[batch_index, :, y:(y+2)] = -1

            # Let's add a second set of missing data bars to give the model an extra challenge!
            x, y = random_location(image_shape, width=2)
            batch_X[batch_index, x:(x+2), :] = -1
            batch_X[batch_index, :, y:(y+2)] = -1

            # Now let's add some larger (beautiful and very square) clouds
            batch_X = insert_random_box(batch_X, batch_index, image_shape, box_width=6, box_class=-1)

        # This changes the shape of batch_Y from
        # (batch_size,) + image_shape[:2] + (1,) to
        # (batch_size,) + image_shape[:2] + (n_classes,)
        batch_Y = to_categorical(batch_Y, num_classes=n_classes)

        # Note: generator returns tuples of (inputs, targets)
        yield (
            batch_X, batch_Y
        )
