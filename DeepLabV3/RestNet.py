#Resnet structure
import numpy as np
import tensorflow as tf
from hyper_param import*




def activation_summary(x):
    '''
    param x : A input tensor
    return : Add histogram summary and scalar summary of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activation',x)
    tf.summary.scalar(tensor_name+'/sparsity',tf.nn.zero_fraction(x))

def create_variables(name, shape, initializer = tf.contrib.layers.xavier_initializer(),is_fc_layer = False):
    '''
    param name : a string , specifying the name of new variable 
    param shape : a list of dimensions 
    param : initializer : the initialization algorithms used for weights  default : Xavier
    params is_fc_layer: flag showing if the variables created is for fc layer 
    '''
    if is_fc_layer:
        regularizer = tf.contrib.layers.l2_regularizer(scale = FLAGS.weight_decay)
    else :
        regularizer = tf.contrib.layers.l2_regularizer(scale = FLAGS.weight_decay)
    
    new_variables = tf.get_variable(name, shape , initializer=initializer, regularizer=regularizer)
    return new_variables

def output_layer (input_layer, num_labels):
    '''
    param input_layer : flattend 2D tensor
    param num_lables: number of classes
    return the output of FC layer : Y =Wx+b
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name = 'fc_weight',shape = [input_dim,num_labels],is_fc_layer = True,initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name = 'fc_bias',shape = [num_labels],is_fc_layer = False,initializer = tf.zeros_initializer())
    output = tf.matmul(input_layer,fc_w) + fc_b
    return output

def batch_norm_layer(input_layer,dimension):
    ''' 
    batch normalization function , batch norm is used right after convolution but before activation
    param input_layer : input 4D tensor 
    param dimension :input_layer.get_shape.as_list()[-1], the depth of 4D tensor
    '''
    mean , variance = tf.nn.moments(input_layer,axes = [0,1,2])
    beta = tf.get_variable(name = 'beta', shape=dimension, dtype=tf.float32,initializer= tf.constant_initializer(0.0,tf.float32))
    gamma = tf.get_variable(name = 'gamma',shape=dimension,dtype=tf.float32, initializer = tf.constant_initializer(1.0,tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer,  mean , variance, beta,gamma,0.001)

    return bn_layer

def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''helper function , building block of structure conv-batch_norm-relu
       param input_layer :4D tensor
       param filter_shape : list , [Height, Width,Depth, Number]
       param stride : stride of convolution
       return output out of stacked layers
    '''
    out_channel = filter_shape[-1] 
    filter = create_variables(name = 'conv',shape = filter_shape)
    conv_out = tf.nn.conv2d(input_layer,filter = filter,strides = [1,stride,stride,1],padding = 'SAME')
    bn_out = batch_norm_layer(conv_out,out_channel)
    relu_out = tf.nn.relu(bn_layer)
    return relu_out

def bn_relu_conv_layer(input_layer,filter_shape,stride,debug = False):
    '''helper function , building block of structure conv-batch_norm-relu
       param input_layer :4D tensor
       param filter_shape : list , [Height, Width,Depth, Number]
       param stride : stride of convolution
       return output out of stacked layers
    '''
    out_channel = input_layer.get_shape().as_list()[-1]
    filter = create_variables(name = 'conv',shape = filter_shape)
    bn_out = batch_norm_layer(input_layer,out_channel)
    relu_out = tf.nn.relu(bn_out)
    conv_out = tf.nn.conv2d(relu_out,filter = filter, strides = [1,stride,stride,1],padding = 'SAME')
    if debug == True :
        dimension = filter.get_shape().as_list()
        print('The input to the conv layer has shape')
        print(dimension)
    
    return conv_out
def bn_conv(input_layer,name_1,filter_shape,use_atrous = False,atrous_rate = None,debug = False):
    out_channel = input_layer.get_shape().as_list()[-1]
    
    filter = create_variables(name = name_1,shape = filter_shape)
    bn_out = batch_norm_layer(input_layer,out_channel)
    if use_atrous == False:
        conv_out = tf.nn.conv2d(bn_out,filter = filter, strides = [1,1,1,1],padding = 'SAME')
    else :
        conv_out = tf.nn.atrous_conv2d(bn_out,filters = filter,rate = atrous_rate,padding = 'SAME')
    if debug == True :
        dimension = filter.get_shape().as_list()
        print('The input to the conv layer has shape')
        print(dimension)
    
    return conv_out

def bn_relu_atrous_conv(input_layer, filter_shape,atrous_rate,debug = False):
    out_channel = input_layer.get_shape().as_list()[-1]
    filter = create_variables(name = 'bn_conv',shape = filter_shape)
    bn_out = batch_norm_layer(input_layer,out_channel)
    relu_out = tf.nn.relu(bn_out)
    conv_out = tf.nn.atrous_conv2d(relu_out,filters = filter,rate = atrous_rate,padding = 'SAME')
    if debug == True :
        dimension = conv_out.get_shape().as_list()[-1]
        print('the dimension of last conv layer is ')
        print(dimension)
    return conv_out

def residual_block(input,out_channel,first_block = False,debug = False):
    '''build a redidual block
       param input : input layer , a 4D tensor 
       param out_channel : output channel depth 
       param first_layer : check if this is the first block of ResNet
       return output of stacked layers 
    '''
    input_channel = input.get_shape().as_list()[-1]

    #check if the input channel is the same as the output channel , if different , we need 
    #to pad zeros to increase dimension 
    if input_channel*2 == out_channel:
        increase_dim = True
        stride = 2
    elif input_channel == out_channel:
        increase_dim = False
        stride = 1
    else :
        raise ValueError('Output and input channel does not match in residual blocks ')

    with tf.variable_scope('conv1_in_block'):
        if first_block:
            #filter = create_variables(name = 'conv',shape =[7,7,input_channel,out_channel])
            #conv1 = tf.nn.conv2d(input,filter = filter,strides = [1,2,2,1],padding = 'SAME')  
            #max_pool_3x3 = tf.nn.maxpool(conv1,ksize=[1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
            #output_1=max_pool_3x3
            filter = create_variables(name = 'conv',shape = [3,3,input_channel,out_channel])
            conv1 = tf.nn.conv2d(input,filter = filter,strides = [1,1,1,1],padding = 'SAME')
            output_1 = conv1
        else :
            #output_1 = conv_bn_relu_layer(input,filter_shape = [3,3,input_channel,out_channel],stride=1)
            output_1 = bn_relu_conv_layer(input,filter_shape = [3,3,input_channel,out_channel],stride=stride,debug = debug)
    with tf.variable_scope('conv2_in_block'):
        #output_2 = conv_bn_relu_layer(output_1,filter_shape = [3,3,input_channel,out_channel],stride = 1)
        output_2 = bn_relu_conv_layer(output_1,filter_shape = [3,3,out_channel,out_channel],stride = 1,debug = debug)
    #print('The shape for output2 is !!!!!!!!!!!!!!!!!')
    #print(output_2.get_shape().as_list())
    #print(increase_dim)

    if increase_dim == True :   
        '''
        in input channel and out channel does not match, it means the input layer
        has been downsampled , so to add identity mapping , need to manually 
        downsample using average pool
        '''
        pooled_input = tf.nn.avg_pool(input, ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
        depth_add = input_channel //2
        padded_input = tf.pad(pooled_input,[[0,0],[0,0],[0,0],[depth_add,depth_add]])
        if debug == True :
            print('the shape of padded input is ')
            print(padded_input.get_shape().as_list())
    else :
        padded_input = input

    final_out = output_2 + padded_input
    return final_out

def residual_atrous_block(input_layer,output_channel,multi_grids,output_stride,debug):
    input_channel = input_layer.get_shape().as_list()[-1]
    #check if the input channel is the same as the output channel , if different , we need 
    #to pad zeros to increase dimension 
    if input_channel*2 == output_channel:
        increase_dim = True
    elif input_channel == output_channel:
        increase_dim = False
    else :
        raise ValueError('Output and input channel does not match in residual blocks ')

    with tf.variable_scope('conv1_in_block'):
        atrous_rate = multi_grids[0]*(32/output_stride)
        output_1 = bn_relu_atrous_conv(input_layer=input_layer,filter_shape = [3,3,input_channel,output_channel],atrous_rate = atrous_rate)
    with tf.variable_scope('conv2_in_block'):
        atrous_rate = multi_grids[1]*(32/output_stride)
        output_2 = bn_relu_atrous_conv(input_layer=output_1,filter_shape = [3,3,output_channel,output_channel],atrous_rate = atrous_rate)
    with tf.variable_scope('conv3_in_block'):
        atrous_rate = multi_grids[2]*(32/output_stride)
        output_3 = bn_relu_atrous_conv(input_layer=output_2,filter_shape = [3,3,output_channel,output_channel],atrous_rate = atrous_rate)

    if increase_dim == True :   
        '''
        in input channel and out channel does not match, it means the input layer
        has been downsampled , so to add identity mapping , need to manually 
        downsample using average pool
        '''
        pooled_input = tf.nn.avg_pool(input_layer, ksize = [1,2,2,1],strides = [1,1,1,1],padding = 'SAME')
        depth_add = input_channel //2
        padded_input = tf.pad(pooled_input,[[0,0],[0,0],[0,0],[depth_add,depth_add]])
        if debug == True :
            print('the shape of padded input is ')
            print(padded_input.get_shape().as_list())
    else :
        padded_input = input_layer

    final_out = output_2 + padded_input
    return final_out




def inference (input_batch,n,reuse,use_atrous= False):
    '''
    main function that definea ResNet
    param :n is the number of residual block for one type of filter
    param reuse :set to true when building validation graph to share the weights from training graph
    '''
    layers = []
    num_classes = 21

    with tf.variable_scope('conv0',reuse = reuse):
        '''first convolutional layer
        '''
        filter= create_variables(name='conv0_f',shape=[3,3,3,16])
        conv0 = tf.nn.conv2d(input_batch,filter=filter,strides = [1,1,1,1],padding = 'SAME')
        activation_summary(conv0)
        layers.append(conv0)
    
    for i in range(n):
        with tf.variable_scope('conv1_%d' %(i+1),reuse = reuse):
            #if i==0 :
            #    conv1 = residual_block(layers[-1],16,debug = False,first_block=False)
            #else :
            conv1 = residual_block(layers[-1],16)
            activation_summary(conv1)
            layers.append(conv1)                                 #[128,32,32,16]
    for i in range(n):
        with tf.variable_scope('conv2_%d'%(i+1),reuse = reuse):
            conv2 = residual_block(layers[-1],32,debug = False)
            activation_summary(conv2)
            layers.append(conv2)
        assert conv2.get_shape().as_list()[1:] == [16,16,32]
    if use_atrous ==True: 
        for i in range(n):
            with tf.variable_scope('atrous_conv%d'%(i+1),reuse = reuse):
                atrous_conv = residual_atrous_block(layers[-1],output_channel = 64,multi_grids = [1,2,4],output_stride=16,debug = False)
                activation_summary(atrous_conv)
                layers.append(atrous_conv)
        #with tf.variable_scope('ASPP',reuse = True):
        input = layers[-1]
        input_channel = input.get_shape().as_list()[-1]
        '''Atrous spatial pyramid pooling a)'''
        with tf.variable_scope('ASPP_conv1',reuse = False):
            conv_1x1 = bn_conv(input_layer = input,name_1 = 'conv1x1_1',filter_shape=[1,1,input_channel,256],use_atrous = False,debug = False)
        with tf.variable_scope('ASPP_conv2',reuse = False):
            conv3x3_1 = bn_conv(input_layer = input, filter_shape = [1,1,input_channel,256],use_atrous = True, atrous_rate = 6, debug = False,name_1 = 'conv3x3_1')
        with tf.variable_scope('ASPP_conv3',reuse = False):
            conv3x3_2 = bn_conv(input_layer = input, filter_shape = [1,1,input_channel,256],use_atrous = True, atrous_rate = 12, debug = False,name_1 = 'conv3x3_2')
        with tf.variable_scope('ASPP_conv4',reuse = False):
            conv3x3_3 = bn_conv(input_layer = input, filter_shape = [1,1,input_channel,256],use_atrous = True, atrous_rate = 18, debug = False,name_1 = 'conv3x3_3')
        '''image-level feature'''
        global_avg_pool = tf.reduce_mean(input,[1,2])
        #gloaal_channel = global_avg_pool.get_shape().as_list()[-1]
        global_filter= create_variables(name='conv0_f',shape=[1,1,input_channel,256])
        global_feat = tf.nn.conv2d(input,filter=global_filter,strides = [1,1,1,1],padding = 'SAME')
        global_feat = batch_norm_layer(global_feat,256)
        '''concatenate'''
        ASPP_out = tf.concat([conv_1x1,conv3x3_1,conv3x3_2,conv3x3_3,global_feat],3)
        '''output , pass through a 1x1 conv with 256 filters and then the final conv layer to produce logits '''
        ASPP_out_channel = ASPP_out.get_shape().as_list()[-1]
        output_aspp_conv = create_variables(name = 'conv_aspp',shape = [1,1,ASPP_out_channel,256])
        output_aspp = tf.nn.conv2d(ASPP_out,filter = output_aspp_conv,strides = [1,1,1,1],padding = 'SAME')
        with tf.variable_scope('ASPP_out',reuse = False):
            output_aspp_channel = output_aspp.get_shape().as_list()[-1]
            output_aspp = batch_norm_layer(output_aspp,output_aspp_channel)
        final_out_conv =  create_variables(name = 'final_aspp_out',shape = [1,1,output_aspp_channel,num_classes])
        final_out = tf.nn.conv2d(output_aspp,filter = final_out_conv,strides = [1,1,1,1],padding = 'SAME')
            
        return final_out

        
    else:
        for i in range(n):
            with tf.variable_scope('conv3_%d'%(i+1),reuse=reuse):
                conv3 = residual_block(layers[-1],64)
                layers.append(conv3)
            assert conv3.get_shape().as_list()[1:] ==[8,8,64]
        with tf.variable_scope('fc',reuse = reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_norm_layer(layers[-1],in_channel)
            relu_out = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_out,[1,2])   #2D
            assert global_pool.get_shape().as_list()[-1:] ==[64]
            output = output_layer(global_pool,10)
            layers.append(output)

    return layers[-1]

def test_graph(train_dir='logs'):
    '''
    test function , look the results on tensorboard
    '''
    input_tensor = tf.constant(np.ones([128,32,32,3]),dtype = tf.float32)
    result = inference(input_tensor,2,reuse = False,use_atrous = True)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(train_dir,sess.graph)

test_graph(train_dir = 'logs')