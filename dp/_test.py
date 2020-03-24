"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

class _test():
	def __init__(self, model_name, model_type, model_netG, model_path, dataset_path, result_path, norm_set):
		self.opt = TestOptions().parse()
		self.opt.num_threads = 0   # test code only supports num_threads = 1
		self.opt.batch_size = 1    # test code only supports batch_size = 1
		self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
		self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
		self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

		self.opt.name = model_name
		self.opt.model = model_type
		self.opt.netG = model_netG
		self.opt.checkpoints_dir = model_path
		self.opt.dataroot = dataset_path
		self.opt.results_dir = result_path
		self.opt.norm = norm_set

		self.model = None
		self.webpage = None

	def setModel(self):
		if self.model == None:
			self.model = create_model(self.opt)      # create a model given opt.model and other options
			self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers

	def setWebpage(self):
		if self.webpage == None:
			web_dir = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.epoch))  # define the website directory
			self.webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opt.name, self.opt.phase, self.opt.epoch))

	def setDataSet(self):
		self.dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options

	def doTest(self):
		self.setModel()
		self.setWebpage()
		self.setDataSet()
		# test with eval mode. This only affects layers like batchnorm and dropout.
		# For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
		# For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
		if self.opt.eval:
			self.model.eval()
		for i, data in enumerate(self.dataset):
			if i >= self.opt.num_test:  # only apply our model to opt.num_test images.
				break
			self.model.set_input(data)  # unpack data from data loader
			self.model.test()           # run inference
			visuals = self.model.get_current_visuals()  # get image results
			img_path = self.model.get_image_paths()     # get image paths
			if i % 5 == 0:  # save images to an HTML file
				print('processing (%04d)-th image... %s' % (i, img_path))
			save_images(self.webpage, visuals, img_path, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize)
		self.webpage.save()  # save the HTML


