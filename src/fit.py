import matplotlib.pyplot as plt
from tensorflow.keras import callbacks

from generator import get_generator
from model import get_model


def main(image_shape=(64, 64, 1), n_classes=3, n_blocks=4, batch_size=20):

    print("hello")

    model = get_model(image_shape, n_blocks, n_classes)

    generator = get_generator(image_shape, n_classes, batch_size)

    X, Y = next(generator)

    for index in range(batch_size):
        plt.imshow(X[index, :, :, 0])
        plt.savefig(f"example_X_{index}.png")
        plt.close()

        plt.imshow(Y[index, :, :, :].argmax(axis=-1))
        plt.savefig(f"example_Y_{index}.png")
        plt.close()

    history = model.fit(
        x=generator,
        steps_per_epoch=50,
        epochs=10,
        verbose=True,
        callbacks=[
            callbacks.EarlyStopping(
                patience=20, monitor="val_loss", restore_best_weights=True, verbose=True
            )
        ],
        validation_data=generator,  # We use the same generator for training and validation
        validation_steps=50,
    )

    predictions = model.predict(X)

    for index in range(batch_size):
        plt.imshow(predictions[index, :, :, :].argmax(axis=-1))
        plt.savefig(f"example_predictions_{index}.png")
        plt.close()


if __name__ == "__main__":
    main()
