import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import datetime
import re
import os


# Load data
(train_ds, val_ds) = tfds.load(name='horses_or_humans', split=['train', 'test'],
                               as_supervised=True, batch_size=32)


def normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return img, label


# Normalize pixels
train_ds = train_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)

# Set checkpointing settings
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(os.getcwd(), 'Saved_Model', 'Models.{epoch}-{val_loss:.2f}.hdf5'),
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode='min',
    save_freq='epoch',
    period=5,
    options=None,
    initial_value_threshold=None,
)


# If model(s) already exists, continue training
if len(os.listdir(os.path.join(os.getcwd(), 'Saved_Model'))) > 1:

    # Regular expression pattern to extract epoch number
    pattern = '[^0-9]+([0-9]+).+'
    filenames = os.listdir(os.path.join(os.getcwd(), 'Saved_Model'))

    # Find epoch number
    max_epoch = -1
    max_epoch_filename = None
    for filename in filenames:
        regexp_match_list = re.findall(pattern=pattern, string=filename)

        # If curr file is log.txt no match
        if len(regexp_match_list) == 0: continue

        last_epoch = int(regexp_match_list[0])
        if last_epoch > max_epoch:
            max_epoch = last_epoch
            max_epoch_filename = filename

    filename = max_epoch_filename

    # Update log file
    date_time = str(datetime.datetime.now())
    date_time = date_time[:len(date_time) - 7]
    with open(file=os.path.join(os.getcwd(), 'Saved_Model', 'log.txt'), mode='a') as f:
        f.write('Training has resumed at: ' + date_time + '\n')
        f.write('Resuming from Epoch Number: ' + str(max_epoch+1) + '\n\n')

    # Load model and continue training model from last epoch
    checkpoint_path = os.path.join(os.getcwd(), 'Saved_Model', filename)
    model = load_model(filepath=checkpoint_path)
    model.fit(x=train_ds, epochs=50, validation_data=val_ds, callbacks=[checkpoint], initial_epoch=max_epoch)

else:
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=(300, 300, 3),
        alpha=1.0,
        minimalistic=False,
        include_top=True,
        weights=None,
        input_tensor=None,
        classes=2,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation='softmax',
    )

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Write to log file
    date_time = str(datetime.datetime.now())
    date_time = date_time[:len(date_time) - 7]
    with open(file=os.path.join(os.getcwd(), 'Saved_Model', 'log.txt'), mode='w') as f:
        f.write('Training has begun at: ' + date_time + '\n\n')

    model.fit(x=train_ds, epochs=50, validation_data=val_ds, callbacks=[checkpoint], initial_epoch=0)
