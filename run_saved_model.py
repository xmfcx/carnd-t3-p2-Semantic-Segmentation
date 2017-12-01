import os.path
import tensorflow as tf
import helper
import scipy
import numpy as np

sess = tf.Session()

path_save_location = './data/saved_model/'
name_model = 'saved_model'

saver = tf.train.import_meta_graph(path_save_location + name_model + '.meta')

saver.restore(sess, tf.train.latest_checkpoint(path_save_location))

operations = sess.graph.get_operations()

graph = tf.get_default_graph()

name_logits = 'Reshape_2:0'
logits = graph.get_tensor_by_name(name_logits)

name_keep_prob = 'keep_prob:0'
keep_prob = graph.get_tensor_by_name(name_keep_prob)

image_input_name = 'image_input:0'
image_input = graph.get_tensor_by_name(image_input_name)

image_shape = (160, 576)
image_path = './data/my_tests/'
image_name = 'umm_000015.png'
image_file = image_path + image_name
image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

image_softmax = sess.run([tf.nn.softmax(logits)],
                         {keep_prob: 1.0, image_input: [image]})

image_softmax = image_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])

image_segment = (image_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)

image_mask = np.dot(image_segment, np.array([[0, 255, 0, 127]]))
image_mask = scipy.misc.toimage(image_mask, mode="RGBA")

image_final_result = scipy.misc.toimage(image)
image_final_result.paste(image_mask, box=None, mask=image_mask)

path_result_images = 'data/my_tests_results'
scipy.misc.imsave(os.path.join(path_result_images, image_name),
                  image_final_result)
