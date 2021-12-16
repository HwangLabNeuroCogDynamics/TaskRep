####################################################################
# Create various representational models
####################################################################
import numpy as np
classes = ['dcb', 'dcr', 'dpb', 'dpr', 'fcb', 'fcr', 'fpb', 'fpr']

def create_RSA_models():
	'''
	###### Create models of representational similarity matrices
	'''
	context_model = np.zeros((8,8))
	shape_model = np.zeros((8,8))
	color_model = np.zeros((8,8))
	identity_model = np.zeros((8,8))
	swapped_dimension_model = np.zeros((8,8))
	nonswapped_dimension_model = np.zeros((8,8))
	swapped_feature_model = np.zeros((8,8))
	nonswapped_feature_model = np.zeros((8,8))
	swapped_task_model = np.zeros((8,8))
	nonswapped_task_model = np.zeros((8,8))

	for it, target in enumerate(classes):
		for ic, confusion in enumerate(classes):
			#context
			if target[0] == confusion[0]: #first letter, solid or hollow
				context_model[it, ic] = 1
			#shape
			if target[1] == confusion[1]:  #second letter, circle or polygon
				shape_model[it, ic] = 1
			#color
			if target[2] == confusion[2]:
				color_model[it, ic] = 1

			if target == confusion:
				identity_model[it, ic] = 1

			if ((target=='dcb' and confusion == 'dcb') or (target=='dcb' and confusion == 'dpb') or (target=='dpb' and confusion == 'dpb') or (target=='dpb' and confusion == 'dcb')):
				swapped_dimension_model[it, ic] = 1
			elif ((target=='dcr' and confusion == 'dcr') or (target=='dcr' and confusion == 'dpr') or (target=='dpr' and confusion == 'dpr') or (target=='dpr' and confusion == 'dcr')):
				swapped_dimension_model[it, ic] = 1
			elif ((target=='fcr' and confusion == 'fcr') or (target=='fcr' and confusion == 'fcb') or (target=='fcb' and confusion == 'fcb') or (target=='fcb' and confusion == 'fcr')):
				swapped_dimension_model[it, ic] = 1
			elif ((target=='fpr' and confusion == 'fpr') or (target=='fpr' and confusion == 'fpb') or (target=='fpb' and confusion == 'fpb') or (target=='fpb' and confusion == 'fpr')):
				swapped_dimension_model[it, ic] = 1

			if ((target=='dcb' and confusion == 'dcb') or (target=='dcb' and confusion == 'dcr') or (target=='dcr' and confusion == 'dcr') or (target=='dcr' and confusion == 'dcb')):
				nonswapped_dimension_model[it, ic] = 1
			elif ((target=='dpb' and confusion == 'dpb') or (target=='dpb' and confusion == 'dpr') or (target=='dpr' and confusion == 'dpr') or (target=='dpr' and confusion == 'dpb')):
				nonswapped_dimension_model[it, ic] = 1
			elif ((target=='fcr' and confusion == 'fcr') or (target=='fcr' and confusion == 'fpr') or (target=='fpr' and confusion == 'fpr') or (target=='fpr' and confusion == 'fcr')):
				nonswapped_dimension_model[it, ic] = 1
			elif ((target=='fcb' and confusion == 'fcb') or (target=='fcb' and confusion == 'fpb') or (target=='fpb' and confusion == 'fpb') or (target=='fpb' and confusion == 'fcb')):
				nonswapped_dimension_model[it, ic] = 1

			if ((target=='dcb') and ((confusion == 'dcr') or (confusion == 'fcr') or (confusion == 'fcb'))):
				swapped_feature_model[it, ic] = 1
				swapped_feature_model[it, it] = 1
			elif ((target=='dcr') and ((confusion == 'dcb') or (confusion == 'fcr') or (confusion == 'fcb'))):
				swapped_feature_model[it, ic] = 1
				swapped_feature_model[it, it] = 1
			elif ((target=='dpr') and ((confusion == 'dpb') or (confusion == 'fpr') or (confusion == 'fpb'))):
				swapped_feature_model[it, ic] = 1
				swapped_feature_model[it, it] = 1
			elif ((target=='dpb') and ((confusion == 'dpr') or (confusion == 'fpr') or (confusion == 'fpb'))):
				swapped_feature_model[it, ic] = 1
				swapped_feature_model[it, it] = 1
			elif ((target=='fcr') and ((confusion == 'fpr') or (confusion == 'dcr') or (confusion == 'dpr'))):
				swapped_feature_model[it, ic] = 1
				swapped_feature_model[it, it] = 1
			elif ((target=='fpr') and ((confusion == 'fcr') or (confusion == 'dcr') or (confusion == 'dpr'))):
				swapped_feature_model[it, ic] = 1
				swapped_feature_model[it, it] = 1
			elif ((target=='fcb') and ((confusion == 'fpb') or (confusion == 'dcb') or (confusion == 'dpb'))):
				swapped_feature_model[it, ic] = 1
				swapped_feature_model[it, it] = 1
			elif ((target=='fpb') and ((confusion == 'fcb') or (confusion == 'dcb') or (confusion == 'dpb'))):
				swapped_feature_model[it, ic] = 1
				swapped_feature_model[it, it] = 1

			if ((target=='dpr') and ((confusion == 'dcr') or (confusion == 'fcr') or (confusion == 'fpr'))):
				nonswapped_feature_model[it, ic] = 1
				nonswapped_feature_model[it, it] = 1
			elif ((target=='dcr') and ((confusion == 'dpr') or (confusion == 'fcr') or (confusion == 'fpr'))):
				nonswapped_feature_model[it, ic] = 1
				nonswapped_feature_model[it, it] = 1
			elif ((target=='dpb') and ((confusion == 'dcb') or (confusion == 'fcb') or (confusion == 'fpb'))):
				nonswapped_feature_model[it, ic] = 1
				nonswapped_feature_model[it, it] = 1
			elif ((target=='dcb') and ((confusion == 'dpb') or (confusion == 'fcb') or (confusion == 'fpb'))):
				nonswapped_feature_model[it, ic] = 1
				nonswapped_feature_model[it, it] = 1
			elif ((target=='fcr') and ((confusion == 'fcb') or (confusion == 'dcr') or (confusion == 'dcb'))):
				nonswapped_feature_model[it, ic] = 1
				nonswapped_feature_model[it, it] = 1
			elif ((target=='fcb') and ((confusion == 'fcr') or (confusion == 'dcr') or (confusion == 'dcb'))):
				nonswapped_feature_model[it, ic] = 1
				nonswapped_feature_model[it, it] = 1
			elif ((target=='fpr') and ((confusion == 'fpb') or (confusion == 'dpr') or (confusion == 'dpb'))):
				nonswapped_feature_model[it, ic] = 1
				nonswapped_feature_model[it, it] = 1
			elif ((target=='fpb') and ((confusion == 'fpr') or (confusion == 'dpr') or (confusion == 'dpb'))):
				nonswapped_feature_model[it, ic] = 1
				nonswapped_feature_model[it, it] = 1

			if ((target=='dcb') and ((confusion == 'dcr') or (confusion == 'fcb') or (confusion == 'fpb'))):
				swapped_task_model[it, ic] = 1
				swapped_task_model[it, it] = 1
			elif ((target=='dcr') and ((confusion == 'dcb') or (confusion == 'fcb') or (confusion == 'fpb'))):
				swapped_task_model[it, ic] = 1
				swapped_task_model[it, it] = 1
			elif ((target=='dpr') and ((confusion == 'dpb') or (confusion == 'fcr') or (confusion == 'fpr'))):
				swapped_task_model[it, ic] = 1
				swapped_task_model[it, it] = 1
			elif ((target=='dpb') and ((confusion == 'dpr') or (confusion == 'fcr') or (confusion == 'fpr'))):
				swapped_task_model[it, ic] = 1
				swapped_task_model[it, it] = 1
			elif ((target=='fcr') and ((confusion == 'fpr') or (confusion == 'dpr') or (confusion == 'dpb'))):
				swapped_task_model[it, ic] = 1
				swapped_task_model[it, it] = 1
			elif ((target=='fpr') and ((confusion == 'fcr') or (confusion == 'dpr') or (confusion == 'dpb'))):
				swapped_task_model[it, ic] = 1
				swapped_task_model[it, it] = 1
			elif ((target=='fcb') and ((confusion == 'fpb') or (confusion == 'dcr') or (confusion == 'dcb'))):
				swapped_task_model[it, ic] = 1
				swapped_task_model[it, it] = 1
			elif ((target=='fpb') and ((confusion == 'fcb') or (confusion == 'dcr') or (confusion == 'dcb'))):
				swapped_task_model[it, ic] = 1
				swapped_task_model[it, it] = 1

			if ((target=='dpr') and ((confusion == 'dcr') or (confusion == 'fpr') or (confusion == 'fpb'))):
				nonswapped_task_model[it, ic] = 1
				nonswapped_task_model[it, it] = 1
			elif ((target=='dcr') and ((confusion == 'dpr') or (confusion == 'fpr') or (confusion == 'fpb'))):
				nonswapped_task_model[it, ic] = 1
				nonswapped_task_model[it, it] = 1
			elif ((target=='dpb') and ((confusion == 'dcb') or (confusion == 'fcr') or (confusion == 'fcb'))):
				nonswapped_task_model[it, ic] = 1
				nonswapped_task_model[it, it] = 1
			elif ((target=='dcb') and ((confusion == 'dpb') or (confusion == 'fcr') or (confusion == 'fcb'))):
				nonswapped_task_model[it, ic] = 1
				nonswapped_task_model[it, it] = 1
			elif ((target=='fcr') and ((confusion == 'fcb') or (confusion == 'dpb') or (confusion == 'dcb'))):
				nonswapped_task_model[it, ic] = 1
				nonswapped_task_model[it, it] = 1
			elif ((target=='fcb') and ((confusion == 'fcr') or (confusion == 'dpb') or (confusion == 'dcb'))):
				nonswapped_task_model[it, ic] = 1
				nonswapped_task_model[it, it] = 1
			elif ((target=='fpr') and ((confusion == 'fpb') or (confusion == 'dpr') or (confusion == 'dcr'))):
				nonswapped_task_model[it, ic] = 1
				nonswapped_task_model[it, it] = 1
			elif ((target=='fpb') and ((confusion == 'fpr') or (confusion == 'dpr') or (confusion == 'dcr'))):
				nonswapped_task_model[it, ic] = 1
				nonswapped_task_model[it, it] = 1

	return context_model, shape_model, color_model, identity_model, swapped_dimension_model, nonswapped_dimension_model, swapped_task_model, nonswapped_task_model, swapped_feature_model, nonswapped_feature_model



#end