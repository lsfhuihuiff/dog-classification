import pickle
from torchvision import transforms
from  torchvision.datasets import ImageFolder

val_imgs = ImageFolder('./dataset/val')

label = val_imgs.class_to_idx
label = {value:key for key,value in label.items()}
print(len(label))
#print(label)

label_hal = open('label.pkl', 'wb')
s = pickle.dumps(label)
label_hal.write(s)
label_hal.close()