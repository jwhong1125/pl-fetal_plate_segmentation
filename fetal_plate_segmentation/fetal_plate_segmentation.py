#!/usr/bin/env python
#
# fetal_plate_segmentation ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#


import os
import sys
import numpy as np
import glob, tempfile

sys.path.append(os.path.dirname(__file__))
from deep_util_JW_predict import *
# import the Chris app superclass
from chrisapp.base import ChrisApp


Gstr_title = """

  __     _        _           _       _                                             _        _   _             
 / _|   | |      | |         | |     | |                                           | |      | | (_)            
| |_ ___| |_ __ _| |    _ __ | | __ _| |_ ___   ___  ___  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
|  _/ _ \ __/ _` | |   | '_ \| |/ _` | __/ _ \ / __|/ _ \/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
| ||  __/ || (_| | |   | |_) | | (_| | ||  __/ \__ \  __/ (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |
|_| \___|\__\__,_|_|   | .__/|_|\__,_|\__\___| |___/\___|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
                 ______| |                 ______         __/ |                                                
                |______|_|                |______|       |___/                                                 

"""

Gstr_synopsis = """

    NAME

       fetal_plate_segmentation.py

    SYNOPSIS

        python fetal_plate_segmentation.py                                         \\
            [-h] [--help]                                               \\
            [-td] [--tempdir]                                           \\
            [-vd] [--verifydir]                                         \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir>

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python fetal_plate_segmentation.py   \\
                                inputdir    outputdir

    DESCRIPTION

        `fetal_plate_segmentation.py` basically does a segment left / right
        fetal cortical plate, and inner region of the cortical plate. 
        This script part of the automatic fetal brain process pipeline at BCH.

    ARGS

        [-h] [--help]
        If specified, show help message and exit.

        [-td] [--tempdir]
        If specified, intermediate result saved at tempdir.

        [-vd] [--verifydir]
        If specified, segmentation result verify image saved at verifydir.

        [--json]
        If specified, show json representation of app and exit.

        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.

        [--savejson <DIR>]
        If specified, save json representation file to DIR and exit.

        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.

        [--version]
        If specified, print version number and exit.

"""


class Fetal_plate_segmentation(ChrisApp):
    """
    An app to segment the cortical plate of fetal T2 MRI using deep leraning..
    """
    AUTHORS                 = 'Jinwoo Hong (Jinwoo.Hong@childrens.harvard.edu)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'A ChRIS plugin app for fetal plate segmentation'
    CATEGORY                = 'Segmentation'
    TYPE                    = 'ds'
    DESCRIPTION             = 'An app to segment the cortical plate of fetal T2 MRI using deep leraning.'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '0.1'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        self.add_argument('-td', '--tempdir',
                           action       = 'store',
                           dest         = 'tempdir', 
                           type         = str, 
                           optional     = True,
                           help         = 'temporay directory path set',
                           default      = '')
        self.add_argument('-vd', '--verifydir',
                           action       = 'store',
                           dest         = 'verifydir', 
                           type         = str, 
                           optional     = True,
                           help         = 'verify directory path set',
                           default      = '')

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())
        # Image list making
        img_list = np.asarray(sorted(glob.glob(options.inputdir+'/Recon_final_nuc.nii')))
        mask = self.SELFPATH+'/mask31_D10.nii.gz'
        # tempdir set
        if options.tempdir == '':
            tempdir = tempfile.mkdtemp()
        else:
            tempdir = options.tempdir

        # Image process # axi
        test_dic, _ = make_dic(img_list, img_list, mask, 'axi', 0)
        model = Unet_network([128,128,1],5, style='basic', ite=3, depth=4).build()
        model.load_weights(self.SELFPATH+'/axi.h5')

        tmask = model.predict(test_dic)
        make_result(tmask,img_list,mask,tempdir+'/','axi')
        tmask = model.predict(test_dic[:,::-1,:,:])
        make_result(tmask[:,::-1,:,:],img_list,mask,tempdir+'/','axi','f1')
        tmask = model.predict(axfliper(test_dic))
        make_result(axfliper(tmask,1),img_list,mask,tempdir+'/','axi','f2')
        tmask = model.predict(axfliper(test_dic[:,::-1,:,:]))
        make_result(axfliper(tmask[:,::-1,:,:],1),img_list,mask,tempdir+'/','axi','f3')

        del model, tmask, test_dic
        reset_graph()
        # Image process # cor
        test_dic, _ =make_dic(img_list, img_list, mask, 'cor', 0)
        model = Unet_network([128,128,1], 5, style='basic',ite=3, depth=4).build()
        model.load_weights(self.SELFPATH+'/cor.h5')

        tmask = model.predict(test_dic)
        make_result(tmask,img_list,mask,tempdir+'/','cor')
        tmask = model.predict(test_dic[:,:,::-1,:])
        make_result(tmask[:,:,::-1,:],img_list,mask,tempdir+'/','cor','f1')
        tmask = model.predict(cofliper(test_dic))
        make_result(cofliper(tmask,1),img_list,mask,tempdir+'/','cor','f2')
        tmask = model.predict(cofliper(test_dic[:,:,::-1,:]))
        make_result(cofliper(tmask[:,:,::-1,:],1),img_list,mask,tempdir+'/','cor','f3')

        del model, tmask, test_dic
        reset_graph()
        # Image process # sag
        test_dic, _ =make_dic(img_list, img_list, mask, 'sag', 0)
        model = Unet_network([128,128,1], 3, style='basic', ite=3, depth=4).build()
        model.load_weights(self.SELFPATH+'/sag.h5')

        tmask = model.predict(test_dic)
        make_result(tmask,img_list,mask,tempdir+'/','sag')
        tmask = model.predict(test_dic[:,::-1,:,:])
        make_result(tmask[:,::-1,:,:],img_list,mask,tempdir+'/','sag','f1')
        tmask = model.predict(test_dic[:,:,::-1,:])
        make_result(tmask[:,:,::-1,:],img_list,mask,tempdir+'/','sag','f2')

        del model, tmask, test_dic
        reset_graph()
        
        make_sum(tempdir+'/*axi*', tempdir+'/*cor*',tempdir+'/*sag*', img_list[0], options.outputdir+'/')
        os.system('rm -rf ' +tempdir)
        
        if options.verifydir != '':
            make_verify(options.inputdir+'/', options.outputdir+'/', options.verifydir+'/')
        

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Fetal_plate_segmentation()
    chris_app.launch()

