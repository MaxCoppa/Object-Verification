# Object-Verification
Given two objects determine whether they represent the same thing or not

Idea : Crop type : crop ==> if we wnat to crop or not the image / for images COCO

To do : 
- Verify no xml / vehicle / car / radar... 
- Improve losses + data augmentation : public and private version
- Change configs too specific only one too much here
- Test one model two public images ... 
- Distances / Predictions ==> Private / put smthg more simple public ? 

utils/image_utils/image_transforms.py ==> Put only simple data augmentation techniques ==> DOne !
utils/model_utils ==> drop too precise from what I ve done ==> Done !
training_evaluation/train_eval.py | training_evaluation/train_veri_model.py==> don't put FR / FA .... ==> Done !
configs ==> too much information 