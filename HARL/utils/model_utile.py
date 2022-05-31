def get_resnet_block_output(model, image, block_ID):
    '''
    Args
    model : encoder model.
    image : tensor of image.
    block_ID : which block we need to use

    Return:
    'original', 'optimizer_weight_decay','optimizer_GD','optimizer_W_GD'
    optimizer.
    '''
    if type(block_ID) == "list":
        list.sort(block_ID)
    else:
        block_ID = [block_ID]

    #0 [<Model_resnet_harry.Conv2dFixedPadding object at 0x0000012055337FA0>,
    #1 <Model_resnet_harry.IdentityLayer object at 0x0000012055337100>,
    #2 <Model_resnet_harry.BatchNormRelu object at 0x0000012054F56940>,
    #3 <keras.layers.pooling.MaxPooling2D object at 0x0000012054F563D0>,
    #4 <Model_resnet_harry.IdentityLayer object at 0x0000012054F56A30>,
    #5 1<Model_resnet_harry.BlockGroup object at 0x0000012054FC4220>,
    #6 2<Model_resnet_harry.BlockGroup object at 0x0000012054F5C940>,
    #7 3<Model_resnet_harry.BlockGroup object at 0x0000012054FA0A00>,
    #8 4<Model_resnet_harry.BlockGroup object at 0x0000012054F6B7C0>]
    
    temp = image
    output_feature = []
    get_id = 0
    for i, layer in enumerate(model.layers):
        temp = model.layers[i](temp)
        if i == block_ID[get_id] + 4:
            output_feature.append(temp)
            get_id += 1
    return output_feature


