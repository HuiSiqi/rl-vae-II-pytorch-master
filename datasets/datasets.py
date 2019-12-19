import glob
import os
from os import path
from PIL import Image
import json
from datetime import datetime
from torchvision.transforms import ToTensor,Compose,Normalize
from torch.utils.data import Dataset
from tqdm import trange, tqdm
import pickle
import random

class IIPDataset(Dataset):
	width = 64
	height = 64

	def __init__(self, root, split,crop_ratio=(0.2,0.2)):
		if split not in ['train', 'test', 'all']:
			raise ValueError

		dir = os.path.join(root, split)
		self.dir = dir
		with open(os.path.join(dir, 'data.json')) as f:
			self._data = json.load(f)
		self._process()
		self.crop_size = {'height':int(crop_ratio[0]*self.height),
		                  'width':int(crop_ratio[1]*self.width)}

	def __len__(self):
		return len(self._data['samples'])

	def __getitem__(self, index):
		gt = self._processed[index]
		j = random.randint(0,self.width-self.crop_size['width'])
		i = random.randint(0,self.height-self.crop_size['height'])
		noisy = gt.clone()
		noisy[:,i:i+self.crop_size['height'],j:j+self.crop_size['width']]=-1
		return gt,noisy,(i,i+self.crop_size['height'],j,j+self.crop_size['width'])

	@staticmethod
	#ratio: the ratio between train and test
	def create(dir,ratio=10):
		from random import shuffle
		filenames = glob.glob(os.path.join(dir, '*.jpg'))
		shuffle(filenames)
		samples = len(filenames)
		test_samples = int(samples/ratio)
		test_file = filenames[:test_samples]
		train_file = filenames[test_samples:]

		def save_file(dir,file):
			if os.path.exists(dir)==False:
				os.makedirs(dir)
			with open(path.join(dir,'data.json'), 'wt') as outfile:
				json.dump(
					{
						'metadata': {
							'num_samples': len(file),
							'time_created': str(datetime.now()),
						},
						'samples': file
					}, outfile, indent=2)

		save_file(os.path.join(dir,'test'),test_file)
		save_file(os.path.join(dir,'train'),train_file)
		save_file(os.path.join(dir,'all'),filenames)

	@staticmethod
	def _process_image(img):
		T = Compose([ToTensor(),Normalize((.5,.5,.5),(.5,.5,.5))])
		return T((img.resize((IIPDataset.width,IIPDataset.height))))

	def _process(self):
		preprocessed_file = os.path.join(self.dir, 'processed.pkl')
		if not os.path.exists(preprocessed_file):
			processed = []
			for sample in tqdm(self._data['samples'], desc='processing data'):
				img = Image.open(sample)

				processed.append(self._process_image(img))

			with open(preprocessed_file, 'wb') as f:
				pickle.dump(processed, f)
			self._processed = processed
		else:
			with open(preprocessed_file, 'rb') as f:
				self._processed = pickle.load(f)






