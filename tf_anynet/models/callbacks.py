import io
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

import tf_anynet.models.turbo_colormap


def plot_to_image(img_data, per_image_standardization=False):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    #import pdb; pdb.set_trace()
    if per_image_standardization:
       img_data = tf.image.per_image_standardization(img_data)
       cmap = 'turbo'
    else:
        cmap = 'turbo'
    plt.imshow(tf.squeeze(img_data), cmap=cmap, interpolation='bilinear') #cmap=plt.cm.binary,
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    plt.close()
    #import pdb; pdb.set_trace()
    # Add the batch dimension
    #image = tf.expand_dims(image, 0)
    return image


class DepthMapImageCallback(keras.callbacks.Callback):
    def __init__(self, eval_data, batch_size, max_outputs, log_dir='.logs/', frequency=1):
        super(DepthMapImageCallback, self).__init__()
        self.frequency = frequency
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir+'/img')
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.max_outputs = max_outputs

    def on_epoch_end(self, epoch, logs=None):
        
        eval_data = self.eval_data.unbatch()

        if self.eval_data and epoch == 0:
            with self.writer.as_default():
                for i, ((imgL,imgR), disp) in eval_data.enumerate():
                    if i < self.max_outputs:
                        tag = f'{i}-input'
                        tf.summary.image(tag, (imgL,imgR), step=epoch)
                        self.writer.flush()
                    else:
                        break
        
        if self.eval_data and epoch % self.frequency == 0:
            samples = tuple(eval_data.as_numpy_iterator())
            #imgs = self.eval_data.map(lambda x,y: x)
            #import pdb; pdb.set_trace()
            preds = self.model.predict(self.eval_data)
            values = preds.values()
            with self.writer.as_default():
                for i in range(0, self.max_outputs):
                    tag = f'{i}-output'

                    imgsplot = [ 
                        plot_to_image(samples[i][1]) ] \
                        + [plot_to_image(v[i])#, per_image_standardization=True) 
                        for v in values
                    ]
                    tf.summary.image(tag, imgsplot, step=epoch, max_outputs=10)
                    self.writer.flush()
