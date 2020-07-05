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
    plt.imshow(tf.squeeze(img_data), cmap=cmap, interpolation='nearest') #cmap=plt.cm.binary,
    
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
    def __init__(self, log_dir='.logs/', frequency=1):
        super(DepthMapImageCallback, self).__init__()
        self.frequency = frequency
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir+'/img')


    def on_epoch_end(self, epoch, logs=None):
        samples = tuple(self.model.eval_samples.as_numpy_iterator())


        if self.model.eval_samples and epoch == 0:
            with self.writer.as_default():
                # for i, data in enumerate(samples):
                    
                #     tag = f'{i}-input'
                #     tf.summary.image(tag,), step=epoch)

                #     self.writer.flush()
                for i, data in enumerate(samples):

                    static =  [ data[0], data[1] ]
                    # static =  [ plot_to_image(samples[i][2])]
                    tag = f'{i}-input'
                    tf.summary.image(tag, static, step=epoch)
                    # tf.summary.image(tag, tf.expand_dims(data[0], axis=0), step=epoch)
                    # tag = f'img-{i}-R'
                    # tf.summary.image(tag, tf.expand_dims(data[1], axis=0), step=epoch)
                    self.writer.flush()


        if self.model.eval_samples and epoch % self.frequency == 0:
            preds = self.model.predict([
                 tf.constant([x[0] for x in samples]),
                 tf.constant([x[1] for x in samples])
                ])
            # nblocks=3, batch_size, H, W, C

            with self.writer.as_default():
                for i in range(0, len(preds[1])):
                    # normalize and encode as a jpg
                    # normalized_pred = tf.image.per_image_standardization(pred)
                    tag = f'{i}-output'
                    #import pdb; pdb.set_trace()
                    # imgs = tf.concat(
                    #     pred + tf.expand_dims([samples[i][0], samples[i][1], samples[i][2]], axis=0)
                    
                    # static =  [ samples[i][0], samples[i][1]]
                    # tf.summary.image(tag, static, step=epoch, max_outputs=10)
                    # self.writer.flush()

                    #tf.summary.image(tag, tf.expand_dims(plot_to_image(samples[i][2]), 0), step=epoch, max_outputs=10)
                    #self.writer.flush()

                    imgsplot = [ plot_to_image(samples[i][2]) ] + [plot_to_image(data, per_image_standardization=True) for data in preds[:,i]] 
                
                    tf.summary.image(tag, imgsplot, step=epoch, max_outputs=10)
                    #for j, data in enumerate(pred):
                        
                        
                    self.writer.flush()

