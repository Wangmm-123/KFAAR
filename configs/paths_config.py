# dataset_paths = {
# 	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
# 	'ffhq': '',
# 	'celeba_test': '',
#
# 	#  Cars Dataset (In the paper: Stanford cars)
# 	'cars_train': '',
# 	'cars_test': '',
#
# 	#  Horse Dataset (In the paper: LSUN Horse)
# 	'horse_train': '',
# 	'horse_test': '',
#
# 	#  Church Dataset (In the paper: LSUN Church)
# 	'church_train': '',
# 	'church_test': '',
#
# 	#  Cats Dataset (In the paper: LSUN Cat)
# 	'cats_train': '',
# 	'cats_test': ''
# }

dataset_paths = {
    'my_train_data': '/root/autodl-tmp/300WLP',
    'my_test_data': '/root/autodl-tmp/AFW_Flip'
}
model_paths = {
	'stylegan_ffhq': 'pretrained_models/300wlp256-002257.pkl',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt'
}
