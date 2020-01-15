class Unet_network:
    def __init__(self, input_shape, out_ch, loss='dice_loss', metrics='dice_coef', style='basic', ite=2, depth=4, dim=32, weights='', init='he_normal',acti='elu',lr=1e-4, dropout=0):
        from keras.layers import Input
        self.style=style
        self.input_shape=input_shape
        self.out_ch=out_ch
        self.loss = loss
        self.metrics = metrics
        self.ite=ite
        self.depth=depth
        self.dim=dim
        self.init=init
        self.acti=acti
        self.weight=weights
        self.dropout=dropout
        self.lr=lr
        self.I = Input(input_shape)

    def conv_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (3,3), padding='same', kernel_initializer=self.init)(x)
        return x

    def conv1_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (1,1), padding='same', kernel_initializer=self.init)(x)
        return x

    def tconv_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2DTranspose, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2DTranspose(dim, 2, strides=2, padding='same', kernel_initializer=self.init)(x)
        return x

    def basic_block(self, inp, dim):
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        return inp

    def res_block(self, inp, dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, Add
        inp2 = inp
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        cb2 = self.conv1_block(inp2,dim)
        return Add()([inp, cb2])

    def dense_block(self, inp, dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, concatenate
        for i in range(self.ite):
            cb = self.conv_block(inp, dim)
            inp = concatenate([inp,cb])
        inp = self.conv1_block(inp,dim)
        return inp

    def RCL_block(self, inp, dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, Add
        RCL=Conv2D(dim, (3,3), padding='same',kernel_initializer=self.init)
        conv=bn()(inp)
        conv=Activation(self.acti)(conv)
        conv=Conv2D(dim,(3,3),padding='same',kernel_initializer=self.init)(conv)
        conv2=bn()(conv)
        conv2=Activation(self.acti)(conv2)
        conv2=RCL(conv2)
        conv2=Add()([conv,conv2])
        for i in range(0, self.ite-2):
            conv2=bn()(conv2)
            conv2=Activation(self.acti)(conv2)
            conv2=Conv2D(dim, (3,3), padding='same',weights=RCL.get_weights())(conv2)
            conv2=Add()([conv,conv2])
        return conv2

    def build_U(self, inp, dim, depth):
        from keras.layers import MaxPooling2D, concatenate, Dropout
        if depth > 0:
            if self.style == 'basic':
                x = self.basic_block(inp, dim)
            elif self.style == 'res':
                x = self.res_block(inp, dim)
            elif self.style == 'dense':
                x = self.dense_block(inp, dim)
            elif self.style == 'RCL':
                x = self.RCL_block(inp, dim)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
            x2 = MaxPooling2D()(x)
            if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            x2 = self.build_U(x2, int(dim*2), depth-1)
            x2 = self.tconv_block(x2,int(dim*2))
            x2 = concatenate([x,x2])
            if self.style == 'basic':
                x2 = self.basic_block(x2, dim)
                if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            elif self.style == 'res':
                x2 = self.res_block(x2, dim)
                if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            elif self.style == 'dense':
                x2 = self.dense_block(x2, dim)
                if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            elif self.style == 'RCL':
                x2 = self.RCL_block(x2, dim)
                if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
        else:
            if self.style == 'basic':
                x2 = self.basic_block(inp, dim)
            elif self.style == 'res':
                x2 = self.res_block(inp, dim)
            elif self.style == 'dense':
                x2 = self.dense_block(inp, dim)
            elif self.style == 'RCL':
                x2 = self.RCL_block(inp, dim)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
        return x2

    def UNet(self):
        from keras.layers import Conv2D
        from keras.models import Model
        from keras.optimizers import Adam
        o = self.build_U(self.I, self.dim, self.depth)
        o = Conv2D(self.out_ch, 1, activation='softmax')(o)
        model = Model(inputs=self.I, outputs=o)
        model.compile(optimizer=Adam(lr=self.lr), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics)])
        if self.weight:
            model.load_weights(self.weight)
        return model

    def build(self):
        return self.UNet()

    def dice_coef(self, y_true, y_pred):
        from keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * K.round(y_pred), axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(K.round(y_pred), axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def dice_loss(self, y_true, y_pred):
        from keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(K.pow(-K.log(dice[1:]),0.3))


def reset_graph():
    from keras import backend as K
    import tensorflow as tf
    K.clear_session()
    tf.reset_default_graph()

def axfliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,:,::-1,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,:,::-1,:]
    return array

def cofliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,::-1,:,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,::-1,:,:]
    return array

def make_dic(img_list, gold_list, mask, dim,flip=0):
    import numpy as np
    import nibabel as nib
    def get_data(img, label, mask_im, mask_min, mask_max, repre_size):
        import nibabel as nib
        img = np.squeeze(nib.load(img).get_data()) * mask_im
        img = img[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]
        loc = np.where(img<np.percentile(img,2))
        img[loc]=0
        loc = np.where(img>np.percentile(img,98))
        img[loc]=0
        loc = np.where(img)
        img[loc] = (img[loc] - np.mean(img[loc])) / np.std(img[loc])
        label = np.squeeze(nib.load(label).get_data())
        label = label[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]
        return img, label

    mask_im = nib.load(mask).get_data()
    mask_min = np.min(np.where(mask_im),axis=1)
    mask_max = np.max(np.where(mask_im),axis=1)
    repre_size = mask_max-mask_min
#     max_repre = np.max(repre_size)
    max_repre = 128
    if dim == 'axi':
        dic = np.zeros([repre_size[2]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[2]*len(img_list), max_repre, max_repre,5])
    elif dim == 'cor':
        dic = np.zeros([repre_size[1]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[1]*len(img_list), max_repre, max_repre,5])
    elif dim == 'sag':
        dic = np.zeros([repre_size[0]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[0]*len(img_list), max_repre, max_repre,3])
    else:
        print('available: axi, cor, sag.   Your: '+dim)
        exit()

    for i in range(0, len(img_list)):
        img, label = get_data(img_list[i], gold_list[i], mask_im, mask_min, mask_max, repre_size)
        if dim == 'axi':
            img2 = np.pad(img,((int((max_repre-img.shape[0])/2),int((max_repre-img.shape[0])/2)),
                               (int((max_repre-img.shape[1])/2),int((max_repre-img.shape[1])/2)),
                               (0,0)), 'constant')
            dic[repre_size[2]*i:repre_size[2]*(i+1),:,:,0]= np.swapaxes(img2,2,0)
            img2 = np.pad(label,((int((max_repre-img.shape[0])/2),int((max_repre-img.shape[0])/2)),
                                 (int((max_repre-img.shape[1])/2),int((max_repre-img.shape[1])/2)),
                                 (0,0)), 'constant')
            img2 = np.swapaxes(img2,2,0)
        elif dim == 'cor':
            img2 = np.pad(img,((int((max_repre-img.shape[0])/2),int((max_repre-img.shape[0])/2)),
                               (0,0),
                               (int((max_repre-img.shape[2])/2),int((max_repre-img.shape[2])/2))),'constant')
            dic[repre_size[1]*i:repre_size[1]*(i+1),:,:,0]= np.swapaxes(img2,1,0)
            img2 = np.pad(label,((int((max_repre-img.shape[0])/2),int((max_repre-img.shape[0])/2)),
                                 (0,0),
                                 (int((max_repre-img.shape[2])/2),int((max_repre-img.shape[2])/2))),'constant')
            img2 = np.swapaxes(img2,1,0)
        elif dim == 'sag':
            img2 = np.pad(img,((0,0),
                              (int((max_repre-img.shape[1])/2),int((max_repre-img.shape[1])/2)),
                              (int((max_repre-img.shape[2])/2),int((max_repre-img.shape[2])/2))), 'constant')
            dic[repre_size[0]*i:repre_size[0]*(i+1),:,:,0]= img2
            img2 = np.pad(label,((0,0),
                                (int((max_repre-img.shape[1])/2),int((max_repre-img.shape[1])/2)),
                                (int((max_repre-img.shape[2])/2),int((max_repre-img.shape[2])/2)),), 'constant')
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
        if (dim == 'axi') | (dim == 'cor'):
            img3 = np.zeros_like(img2)
            back_loc = np.where(img2<0.5)
            left_in_loc = np.where((img2>160.5)&(img2<161.5))
            right_in_loc = np.where((img2>159.5)&(img2<160.5))
            left_plate_loc = np.where((img2>0.5)&(img2<1.5))
            right_plate_loc = np.where((img2>41.5)&(img2<42.5))
            img3[back_loc]=1
        elif dim == 'sag':
            img3 = np.zeros_like(img2)
            back_loc = np.where(img<0.5)
            in_loc = np.where((img2>160.5)&(img2<161.5)|(img2>159.5)&(img2<160.5))
            plate_loc = np.where((img2>0.5)&(img2<1.5)|(img2>41.5)&(img2<42.5))
            img3[back_loc]=1
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

        if dim == 'axi':
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,4]=img3
            img3[:]=0
        elif dim == 'cor':
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,4]=img3
            img3[:]=0
        elif dim == 'sag':
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[in_loc]=1
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[plate_loc]=1
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,2]=img3
            img3[:]=0
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    if flip:
        if dim == 'axi':
            dic=np.concatenate((dic,dic[:,::-1,:,:]),axis=0)
            seg=np.concatenate((seg,seg[:,::-1,:,:]),axis=0)
            dic=np.concatenate((dic, axfliper(dic)),axis=0)
            seg=np.concatenate((seg, axfliper(seg, 1)),axis=0)
        elif dim == 'cor':
            dic=np.concatenate((dic,dic[:,:,::-1,:]),axis=0)
            seg=np.concatenate((seg,seg[:,:,::-1,:]),axis=0)
            dic=np.concatenate((dic, cofliper(dic)),axis=0)
            seg=np.concatenate((seg, cofliper(seg, 1)),axis=0)
        elif dim == 'sag':
            dic = np.concatenate((dic, dic[:,:,::-1,:], dic[:,::-1,:,:]),axis=0)
            seg = np.concatenate((seg, seg[:,:,::-1,:], seg[:,::-1,:,:]),axis=0)
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    return dic, seg


def make_result(tmask, img_list, mask,result_loc,axis,ext=''):
    import nibabel as nib
    import numpy as np
    mask_im = nib.load(mask).get_data()
    mask_min = np.min(np.where(mask_im),axis=1)
    mask_max = np.max(np.where(mask_im),axis=1)
    repre_size = mask_max-mask_min
#     max_repre = np.max(repre_size)
    max_repre = 128

    if np.shape(img_list):
        for i2 in range(len(img_list)):
            print('filename : '+img_list[i2])
            img = nib.load(img_list[i2])
            img_data = np.squeeze(img.get_data())
            img2=img_data[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]
            pr4=tmask[i2*(np.int(tmask.shape[0]/len(img_list))):(i2+1)*(np.int(tmask.shape[0]/len(img_list)))]
            if axis == 'axi':
                pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,2)
                pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,:]
            elif axis == 'cor':
                pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,1)
                pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
            elif axis == 'sag':
                pr4=np.argmax(pr4,axis=3).astype(np.int)
                pr4=pr4[:,:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
            else:
                print('available: axi, cor, sag.   Your: '+axis)
                exit()

            img_data[:] = 0
            img_data[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]=pr4
            new_img = nib.Nifti1Image(img_data, img.affine, img.header)
            filename=img_list[i2].split('/')[-1:][0].split('.')[0]
            filename=filename.split('_')
            if axis== 'axi':
                filename='seg_axi'+ext+'.nii.gz'
            elif axis== 'cor':
                filename='seg_cor'+ext+'.nii.gz'
            elif axis== 'sag':
                filename='seg_sag'+ext+'.nii.gz'
            else:
                print('available: axi, cor, sag.   Your: '+axis)
                exit()
            print('save result : '+result_loc+filename)
            nib.save(new_img, result_loc+str(filename))
    else:
        print('filename : '+img_list)
        img = nib.load(img_list)
        img_data = np.squeeze(img.get_data())
        img2=img_data[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]
        pr4 = tmask
        if axis == 'axi':
            pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,2)
            pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,:]
        elif axis == 'cor':
            pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,1)
            pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
        elif axis == 'sag':
            pr4=np.argmax(pr4,axis=3).astype(np.int)
            pr4=pr4[:,:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
        else:
            print('available: axi, cor, sag.   Your: '+axis)
            exit()

        img_data[:] = 0
        img_data[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]=pr4
        new_img = nib.Nifti1Image(img_data, img.affine, img.header)
        filename=img_list.split('/')[-1:][0].split('.')[0]
        filename=filename.split('_')
        if axis== 'axi':
            filename='seg_axi'+ext+'.nii.gz'
        elif axis== 'cor':
            filename='seg_cor'+ext+'.nii.gz'
        elif axis== 'sag':
            filename='seg_sag'+ext+'.nii.gz'
        else:
            print('available: axi, cor, sag.   Your: '+axis)
            exit()

        print('save result : '+result_loc+filename)
        nib.save(new_img, result_loc+str(filename))

    return 1

def make_sum(axi_filter, cor_filter, sag_filter, input_name, result_loc):
    import nibabel as nib
    import numpy as np
    import sys, glob

    # 1-->axi 2-->cor 3-->sag
    axi_list = sorted(glob.glob(axi_filter))
    cor_list = sorted(glob.glob(cor_filter))
    sag_list = sorted(glob.glob(sag_filter))
    axi = nib.load(axi_list[0])
    cor = nib.load(cor_list[0])
    sag = nib.load(sag_list[0])

    bak = np.zeros(np.shape(axi.get_data()))
    left_in = np.zeros(np.shape(axi.get_data()))
    right_in = np.zeros(np.shape(axi.get_data()))
    left_plate = np.zeros(np.shape(axi.get_data()))
    right_plate = np.zeros(np.shape(axi.get_data()))
    total = np.zeros(np.shape(axi.get_data()))

    for i in range(len(axi_list)):
        axi_data = nib.load(axi_list[i]).get_data()
        cor_data = nib.load(cor_list[i]).get_data()
        if len(sag_list) > i:
            sag_data = nib.load(sag_list[i]).get_data()

        loc = np.where(axi_data==0)
        bak[loc]=bak[loc]+1
        loc = np.where(cor_data==0)
        bak[loc]=bak[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==0)
            bak[loc]=bak[loc]+1

        loc = np.where(axi_data==1)
        left_in[loc]=left_in[loc]+1
        loc = np.where(cor_data==1)
        left_in[loc]=left_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            left_in[loc]=left_in[loc]+1

        loc = np.where(axi_data==2)
        right_in[loc]=right_in[loc]+1
        loc = np.where(cor_data==2)
        right_in[loc]=right_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            right_in[loc]=right_in[loc]+1

        loc = np.where(axi_data==3)
        left_plate[loc]=left_plate[loc]+1
        loc = np.where(cor_data==3)
        left_plate[loc]=left_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            left_plate[loc]=left_plate[loc]+1

        loc = np.where(axi_data==4)
        right_plate[loc]=right_plate[loc]+1
        loc = np.where(cor_data==4)
        right_plate[loc]=right_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            right_plate[loc]=right_plate[loc]+1

    result = np.concatenate((bak[np.newaxis,:], left_in[np.newaxis,:], right_in[np.newaxis,:], left_plate[np.newaxis,:], right_plate[np.newaxis,:]),axis=0)
    result = np.argmax(result, axis=0)
    filename='Segmentation_final.nii'
    new_img = nib.Nifti1Image(result, axi.affine, axi.header)
    nib.save(new_img, result_loc+'/'+filename)

def make_verify(input_dir, label_dir, result_loc):
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import sys

    img = nib.load(input_dir+'/Recon_final_nuc.nii').get_data()
    label = nib.load(label_dir+'/Segmentation_final.nii').get_data()

    f,axarr = plt.subplots(3,3)
    f.patch.set_facecolor('k')

    f.text(0.4, 0.95, label_dir+'/Segmentation_final.nii', size="small", color="White")

    axarr[0,0].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.4)]),cmap='gray')
    axarr[0,0].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.4)]),alpha=0.3, cmap='gnuplot2')
    axarr[0,0].axis('off')

    axarr[0,1].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.5)]),cmap='gray')
    axarr[0,1].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.5)]),alpha=0.3, cmap='gnuplot2')
    axarr[0,1].axis('off')

    axarr[0,2].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.6)]),cmap='gray')
    axarr[0,2].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.6)]),alpha=0.3, cmap='gnuplot2')
    axarr[0,2].axis('off')

    axarr[1,0].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.4),:]),cmap='gray')
    axarr[1,0].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.4),:]),alpha=0.3, cmap='gnuplot2')
    axarr[1,0].axis('off')

    axarr[1,1].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.5),:]),cmap='gray')
    axarr[1,1].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.5),:]),alpha=0.3, cmap='gnuplot2')
    axarr[1,1].axis('off')

    axarr[1,2].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.6),:]),cmap='gray')
    axarr[1,2].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.6),:]),alpha=0.3, cmap='gnuplot2')
    axarr[1,2].axis('off')

    axarr[2,0].imshow(np.rot90(img[np.int(img.shape[0]*0.4),:,:]),cmap='gray')
    axarr[2,0].imshow(np.rot90(label[np.int(label.shape[0]*0.4),:,:]),alpha=0.3, cmap='gnuplot2')
    axarr[2,0].axis('off')

    axarr[2,1].imshow(np.rot90(img[np.int(img.shape[0]*0.5),:,:]),cmap='gray')
    axarr[2,1].imshow(np.rot90(label[np.int(label.shape[0]*0.5),:,:]),alpha=0.3, cmap='gnuplot2')
    axarr[2,1].axis('off')

    axarr[2,2].imshow(np.rot90(img[np.int(img.shape[0]*0.6),:,:]),cmap='gray')
    axarr[2,2].imshow(np.rot90(label[np.int(label.shape[0]*0.6),:,:]),alpha=0.3, cmap='gnuplot2')
    axarr[2,2].axis('off')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(result_loc+'/segmentation_verify.png', facecolor=f.get_facecolor())
    return 0

