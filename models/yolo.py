import configparser
import io
import os
from collections import defaultdict
import threading
import time
import queue
import cv2 

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, Input, ZeroPadding2D, Add,
                          UpSampling2D, MaxPooling2D, Concatenate)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow import convert_to_tensor

from IPython.display import display
from ipywidgets import IntProgress, FloatProgress, widgets

class YOLOv3():
    def __init__(self):
        self.labels = np.array(["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])
        self.anchors = tf.constant([[(116,90), (156,198), (373,326)], [(30,61), (62,45), (59,119)], [(10,13), (16,30), (33,23)]],dtype='float32')
        self.n_labels = len(self.labels)
        self.model_image_w = 608
        self.model_image_h = 608

        self.prediction_threshold = 0.6
        self.max_boxes = 100 
        self.iou_threshold = 0.3

        self.input_buffer_size = 5
        self.output_buffer_size = 30
          
    def run_threaded(self, source = 'webcam', buffer_viz = False):
        self.run_flag = True
        self.in_que = queue.Queue(maxsize=self.input_buffer_size)
        self.out_que = queue.Queue(maxsize=self.output_buffer_size)
        self.in_progress = IntProgress(min=0, max=self.input_buffer_size)
        self.out_progress = IntProgress(min=0, max=self.output_buffer_size)
        
        self.infer_t = 0
        self.render_t = 0
        

        if (buffer_viz):
            print('Input Buffer')
            display(self.in_progress)
            print('Output Buffer')
            display(self.out_progress)
            self.display_infer_t = display("Inference Time (s): ", display_id = True)
            self.display_render_t = display("Render Time (s): ", display_id = True)


        if source == 'webcam': source = 0

        thread1 = threading.Thread(target=self._read, args = (source, ))
        thread2 = threading.Thread(target=self._infer)
        thread3 = threading.Thread(target=self._view)
        #display(progress)
        thread1.start()
        thread2.start()
        thread3.start()

        
        
    def stop(self):
        self.run_flag = False
    
    def _read(self, source = 0):
        cap = cv2.VideoCapture(source)
        ret, img = cap.read()

        if ret == False:
            print('Can\'t read video from source')
            self.run_flag=False
            return

        self.image_h, self.image_w = img.shape[:2]

        self.output_rescale = np.array([self.image_h/self.model_image_h, self.image_w/self.model_image_w, 
                                        self.image_h/self.model_image_h, self.image_w/self.model_image_w])

        while(ret and self.run_flag):
            img_resize = cv2.resize(img, (self.model_image_w, self.model_image_h))
            
            if source == 0:
                img = cv2.flip(img, 1)
                img_resize = cv2.flip(img_resize, 1)

            self.in_que.put((img, convert_to_tensor(img_resize, dtype='float32')[None, ...]/255.0))
            self.in_progress.value += 1

            ret, img = cap.read()

        self.run_flag = False
        cap.release()
        print('Read Ended')

    def _infer(self):
        while(self.run_flag):
            try:
                img, img_resize = self.in_que.get(timeout = 0.5)
            except:
                continue

            self.in_progress.value -= 1
            t1 = time.time()
            yhat = self.model(img_resize, training=False)
            output = self._find_box(yhat)

            self.out_que.put((img, output[:,:-1].numpy().astype('int64'), output[:,-1].numpy()))
            self.out_progress.value+=1

            self.infer_t = self.infer_t*0.9 + (time.time() - t1)*0.1
            self.display_infer_t.update("Inference Time (s): {:.3f}".format(self.infer_t))
        
        print('Inference Ended')
        self.run_flag=False
    
    
    def _view(self):
        while(self.run_flag):
            try:
                img, output, score = self.out_que.get(timeout=0.5)
            except:
                continue
            
            self.out_progress.value -= 1
            t1 = time.time()
            for i,scr in zip(output,score):
                rect = (i[0:4]*self.output_rescale).astype('int64')
                img = cv2.rectangle(img, tuple(rect[1::-1]), tuple(rect[3:1:-1]), (0, 0, 200), 2)
                img = cv2.putText(img, self.labels[i[4]] + ': ' + '{:.3f}'.format(scr), (rect[1], rect[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 1)
            cv2.imshow(f'Video. Press \'e\' to exit.', img)
            
            if cv2.waitKey(1) == ord('e'): break
            
            self.render_t = 0.9*self.render_t + 0.1*(time.time() - t1)
            self.display_render_t.update("Render Time (s): {:.3f}".format(self.render_t))
        cv2.destroyAllWindows()
        self.run_flag=False
        print('View Ended')
    
    @tf.function
    def _find_box(self, preds, batch_id=0):
        def sigmoid(x):
            return 1/(1+tf.exp(-x)) 
    
        def find_all_box_channel(pred, n, batch_id):
            shp = tf.shape(pred)
            div = tf.convert_to_tensor([[self.model_image_h/shp[1],self.model_image_w/shp[2]]],dtype='float32')
            pred = tf.reshape(pred[batch_id],(shp[1],shp[2],-1,self.n_labels+5))

            p_object = sigmoid(pred[...,4:5])
            p_label = sigmoid(pred[...,5:])
            p_cell_label = (p_object*p_label)
            valid = tf.reduce_any(p_cell_label > self.prediction_threshold, axis=-1)

            c_x_y = sigmoid(pred[valid][:,:2])
            c_w_h = tf.exp(pred[valid][:,2:4])
            indices = tf.where(valid)
            temp_x_y = c_x_y + tf.cast(indices[:,:2], 'float32')*div
            temp_h_w = c_w_h[:,::-1]*tf.gather(self.anchors[n,:,::-1],indices[:,-1])/2

            x_y_1 = temp_x_y - temp_h_w
            x_y_2 = temp_x_y + temp_h_w
            p_box_label = p_label[valid]
            boxes = tf.concat((x_y_1,x_y_2, p_box_label), axis=-1)
            return boxes

        def nms(boxes):
            col_indices = tf.cast(tf.argmax(boxes[:,4:],axis=-1), 'int32')
            row_indices = tf.range(tf.shape(boxes)[0])
            indices = tf.stack((row_indices,col_indices+4), axis=1)
            score = tf.gather_nd(boxes, indices)
    
            nms = tf.image.non_max_suppression(boxes[:,0:4], score, self.max_boxes, self.iou_threshold)

            coord = tf.gather(boxes[:,0:4],nms,axis=0)
            score = tf.gather(score,nms,axis=0)[:,None]
            col_indices = tf.gather(tf.cast(col_indices, 'float32'),nms,axis=0)[:,None]
            return tf.concat((coord,col_indices,score),axis=-1)

        collect_all_channels = tf.concat([find_all_box_channel(pred,i, batch_id) for i,pred in enumerate(preds)],axis=0)
        return nms(collect_all_channels)
    
    def _unique_config_sections(self, config_file):
        """Convert all config sections to have unique names.
        Adds unique suffixes to config sections for compability with configparser.
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(config_file) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream
    
    def load_model(self, config_path, weights_path='', output_path='', verbose=0):
        """Load YOLO models. Model should either be in .h5 format or should be given as config file and model weights.
        If output_path is given the function outputs an .h5 file for faster load in the future"""
        if config_path.endswith('.h5'):
            self.model = load_model(config_path, compile=False)
            return
        else:
            assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
                config_path)
            assert weights_path.endswith(
                '.weights'), '{} is not a .weights file'.format(weights_path)

        # Load weights and config.
        if verbose==1: print('Loading weights.')
        weights_file = open(weights_path, 'rb')
        major, minor, revision = np.ndarray(
            shape=(3, ), dtype='int32', buffer=weights_file.read(12))
        if (major*10+minor)>=2 and major<1000 and minor<1000:
            seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
        else:
            seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
        if verbose==1: print('Weights Header: ', major, minor, revision, seen)

        if verbose==1: print('Parsing Darknet config.')
        unique_config_file = self._unique_config_sections(config_path)
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read_file(unique_config_file)

        if verbose==1: print('Creating Keras model.')
        input_layer = Input(shape=(None, None, 3))
        prev_layer = input_layer
        all_layers = []

        weight_decay = float(cfg_parser['net_0']['decay']
                             ) if 'net_0' in cfg_parser.sections() else 5e-4
        count = 0
        out_index = []
        for section in cfg_parser.sections():
            if verbose==1: print('Parsing section {}'.format(section))
            if section.startswith('convolutional'):
                filters = int(cfg_parser[section]['filters'])
                size = int(cfg_parser[section]['size'])
                stride = int(cfg_parser[section]['stride'])
                pad = int(cfg_parser[section]['pad'])
                activation = cfg_parser[section]['activation']
                batch_normalize = 'batch_normalize' in cfg_parser[section]

                padding = 'same' if pad == 1 and stride == 1 else 'valid'

                # Setting weights.
                # Darknet serializes convolutional weights as:
                # [bias/beta, [gamma, mean, variance], conv_weights]
                prev_layer_shape = K.int_shape(prev_layer)

                weights_shape = (size, size, prev_layer_shape[-1], filters)
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size = np.product(weights_shape)

                if verbose==1: print('conv2d', 'bn'
                                        if batch_normalize else '  ', activation, weights_shape)

                conv_bias = np.ndarray(
                    shape=(filters, ),
                    dtype='float32',
                    buffer=weights_file.read(filters * 4))
                count += filters

                if batch_normalize:
                    bn_weights = np.ndarray(
                        shape=(3, filters),
                        dtype='float32',
                        buffer=weights_file.read(filters * 12))
                    count += 3 * filters

                    bn_weight_list = [
                        bn_weights[0],  # scale gamma
                        conv_bias,  # shift beta
                        bn_weights[1],  # running mean
                        bn_weights[2]  # running var
                    ]

                conv_weights = np.ndarray(
                    shape=darknet_w_shape,
                    dtype='float32',
                    buffer=weights_file.read(weights_size * 4))
                count += weights_size

                # DarkNet conv_weights are serialized Caffe-style:
                # (out_dim, in_dim, height, width)
                # We would like to set these to Tensorflow order:
                # (height, width, in_dim, out_dim)
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                conv_weights = [conv_weights] if batch_normalize else [
                    conv_weights, conv_bias
                ]

                # Handle activation.
                act_fn = None
                if activation == 'leaky':
                    pass  # Add advanced activation later.
                elif activation != 'linear':
                    raise ValueError(
                        'Unknown activation function `{}` in section {}'.format(
                            activation, section))

                # Create Conv2D layer
                if stride>1:
                    # Darknet uses left and top padding instead of 'same' mode
                    prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
                conv_layer = (Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding))(prev_layer)

                if batch_normalize:
                    conv_layer = (BatchNormalization(
                        weights=bn_weight_list))(conv_layer)
                prev_layer = conv_layer

                if activation == 'linear':
                    all_layers.append(prev_layer)
                elif activation == 'leaky':
                    act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                    prev_layer = act_layer
                    all_layers.append(act_layer)

            elif section.startswith('route'):
                ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
                layers = [all_layers[i] for i in ids]
                if len(layers) > 1:
                    if verbose==1: print('Concatenating route layers:', layers)
                    concatenate_layer = Concatenate()(layers)
                    all_layers.append(concatenate_layer)
                    prev_layer = concatenate_layer
                else:
                    skip_layer = layers[0]  # only one layer to route
                    all_layers.append(skip_layer)
                    prev_layer = skip_layer

            elif section.startswith('maxpool'):
                size = int(cfg_parser[section]['size'])
                stride = int(cfg_parser[section]['stride'])
                all_layers.append(
                    MaxPooling2D(
                        pool_size=(size, size),
                        strides=(stride, stride),
                        padding='same')(prev_layer))
                prev_layer = all_layers[-1]

            elif section.startswith('shortcut'):
                index = int(cfg_parser[section]['from'])
                activation = cfg_parser[section]['activation']
                assert activation == 'linear', 'Only linear activation supported.'
                all_layers.append(Add()([all_layers[index], prev_layer]))
                prev_layer = all_layers[-1]

            elif section.startswith('upsample'):
                stride = int(cfg_parser[section]['stride'])
                assert stride == 2, 'Only stride=2 supported.'
                all_layers.append(UpSampling2D(stride)(prev_layer))
                prev_layer = all_layers[-1]

            elif section.startswith('yolo'):
                out_index.append(len(all_layers)-1)
                all_layers.append(None)
                prev_layer = all_layers[-1]

            elif section.startswith('net'):
                pass

            else:
                raise ValueError(
                    'Unsupported section header type: {}'.format(section))

        # Create and save model.
        if len(out_index)==0: out_index.append(len(all_layers)-1)
        self.model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
        if verbose==1: print(model.summary())

        if output_path.endswith('.h5'):
            self.model.save('{}'.format(output_path))
            if verbose==1: print('Saved Keras model to {}'.format(output_path))

        # Check to see if all weights have been read.
        remaining_weights = len(weights_file.read()) / 4
        weights_file.close()
        if verbose==1: print('Read {} of {} from Darknet weights.'.format(count, count +
                                                           remaining_weights))
        if remaining_weights > 0:
            print('Warning: {} unused model weights'.format(remaining_weights))