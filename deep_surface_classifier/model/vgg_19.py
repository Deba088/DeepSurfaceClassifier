import os

from tensorflow import keras
from keras import layers

from ..utils import data_generator


def vgg_19_model(train_data, test_data, model_weights_dir, dataset_dir, batch_size, epochs, class_names,
                 num_classes, num_layers_to_unfreeze, use_weight=False):
    img_height = 224
    img_weight = 224

    train_generator = data_generator(train_data, dataset_dir, batch_size, img_height, img_weight,
                                     epochs, class_names, num_classes)
    test_generator = data_generator(test_data, dataset_dir, batch_size, img_height, img_weight,
                                    epochs, class_names, num_classes)

    base_model = keras.applications.VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(img_height, img_weight, 3)
    )

    model_path = f"{model_weights_dir}/vgg19.h5"
    if os.path.exists(model_path) and use_weight:
        base_model.load_weights(model_path)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    if num_layers_to_unfreeze != 0:
        for layer in base_model.layers:
            layer.trainable = False

        for layer in model.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True

    optimizer = keras.optimizers.SGD(
        learning_rate=0.001,
        momentum=0.9
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_data) // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_data) // batch_size
    )

    model.save(model_path)
    return model, history, img_height, img_weight
