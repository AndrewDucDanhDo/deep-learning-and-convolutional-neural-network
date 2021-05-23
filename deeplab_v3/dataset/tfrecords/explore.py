import tensorflow as tf
from tensorflow.data import Dataset, TFRecordDataset
filename = r'D:\Code\DL_CNN\new\deep-learning-and-convolutional-neural-network\deeplab_v3\dataset\tfrecords\validation.tfrecords'
raw_dataset = tf.data.TFRecordDataset(filename)
print(raw_dataset)
# print('x:', tf.io.parse_tensor(x_bytes, out_type=tf.uint8))

# for raw_record in raw_dataset.take(1):
#   example = tf.train.Example()
#   example.ParseFromString(raw_record.numpy().decode('utf-8'))
#   print(example)

ds_2 = TFRecordDataset(TFRecordDataset(filename))
for x in ds_2:
    print(x)