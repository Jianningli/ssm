import nrrd
import gc
import ants, shutil
from glob import glob
from sklearn.decomposition import PCA



class skullRecSSM(object):
	def __init__(self,numOfImg4SSM=30):
		self.numOfImg4SSM=numOfImg4SSM

	def reg(self,fix_img,moving_img):
	    outs = ants.registration(fix_img, moving_img, type_of_transforme = 'Similarity')
	    warped_img = outs['warpedmovout']
	    return warped_img


	def inverse_reg(self,fixed1,moving1,moving2):
		outs1 = ants.registration(fixed1, moving1, type_of_transforme = 'Similarity')
		outs2 = ants.apply_transforms(moving1, moving2,transformlist=outs1['invtransforms'])
		outs2=outs2.numpy()
		outs2=(outs2>0)
		outs2=outs2+1-1
		return outs2



	def ssm_train(self,warped_train_dir):
		complete = glob(warped_train_dir+'*.nrrd')
		pca = PCA(n_components=len(complete))
		#(512,512,222) is the size of the fixed image
		data=np.zeros(shape=(len(complete),512,512,222),dtype='int16')
		for i in range(len(complete)):
			temp,header=nrrd.read(complete[i])
			data[i,:,:,:]=temp
			del temp
		data=data[0:self.numOfImg4SSM]
		self.mean_shape=data.mean(axis=0)
		data=np.reshape(data,(len(complete),512*512*222))
		data_pca = pca.fit_transform(data)

		#explained_variance_ratio
		#percentage=pca.explained_variance_ratio_
		#components_
		#components_=pca.components_

		data_inv=np.linalg.pinv(data)

		del data
		gc.collect()

		self.eigenvec=data_inv.dot(data_pca)

		del data_
		del data_inv
		gc.collect()

	




	def ssm_test(self, testImg, useOnlyMeanShape=False):

		testdata=np.reshape(testImg,(1,512*512*222))
		testdatapca=testdata.dot(self.eigenvec)

		lambda_n=[]
		for i in range(len(testdatapca)):
			lambda_n.append(testdatapca[i])	
		lambda_n=np.array(lambda_n)
		#scale [0,1]
		lambda_n = (lambda_n - np. min(lambda_n))/np. ptp(lambda_n)
		lambda_n=np.transpose(lambda_n)
		reconstructed=eigenvec.dot(lambda_n)
		reconstructed=np.reshape(reconstructed,(512,512,222))
		if useOnlyMeanShape:
			rec=self.mean_shape
		else:
			rec=reconstructed+self.mean_shape

		rec=(rec>0)
		rec=rec+1-1

		return  rec




if __name__ == "__main__":

	fixed_img=ants.image_read('./fixed/001.nrrd')
	moving_train_dir='./moving_train/'
	moving_test_dir='./moving_test/'
	warped_train_dir='./warped_img/train/'
	warped_test_dir='./warped_img/test/'
	results_dir='./results/'


	moving_train_imgs = glob(moving_train_dir+'*.nrrd')
	moving_test_imgs = glob(moving_test_dir+'*.nrrd')
	warped_test_img = glob(warped_test_dir+'*.nrrd')


	model=skullRecSSM(30)

	print('warpping training images...')


	for idx in range(len(moving_train_imgs)):
		NamePrefix = str(idx).zfill(3)
	    moving_img = ants.image_read(moving_train_imgs[idx])
	    outs = model.reg(fixed_img, moving_img)
	    warped_img = outs['warpedmovout']
	    ants.image_write(warped_img, warped_train_dir + NamePrefix +'.nrrd')

	rint('building SSM...')
	model.ssm_train(warped_train_dir)


	print('warpping test images...')
	for idx in range(len(moving_test_imgs)):
		NamePrefix = str(idx).zfill(3)
	    moving_img = ants.image_read(moving_test_imgs[idx])
	    outs = ssm.reg(fixed_img, moving_img)
	    warped_img = outs['warpedmovout']
	    ants.image_write(warped_img, warped_test_dir + NamePrefix +'.nrrd')



	print('fitting...')
	for i in range(len(warped_test_img)):
		NamePrefix = str(i).zfill(3)
		test,h=nrrd.read(warped_test_img[i])
		h['type']='int32'
		h['encoding']='gzip'	
		rec=model.ssm_test(test,useOnlyMeanShape=False)
		implant=rec-test
		nrrd.write(results_dir+'skulls/'+NamePrefix,rec.astype('int32'),h)
		nrrd.write(results_dir+'implants/'+NamePrefix,implant.astype('int32'),h)


	print('converting the results back to original image space...')
	skull_imgs = glob(results_dir+'skulls/'+'*.nrrd')
	implant_imgs = glob(results_dir+'implants/'+'*.nrrd')

	for i in range(len(warped_test_img)):
	    NamePrefix = str(i).zfill(3)
	    moving = ants.image_read(moving_test_imgs[i])
	    moving_results = ants.image_read(implant_imgs[i])
	    converted_img=model.inverse_reg(fixed_img,moving,moving_results)
	    _,header=nrrd.read(moving_test_imgs[i])
	    nrrd.write(implant_imgs,converted_img.astype('int32'),header)

