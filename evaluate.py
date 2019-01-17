import pickle
from utils import *
from PIL import Image
import matplotlib.pyplot as plt

weights_file = "./weights.pkl"
params, cost = pickle.load(open(weights_file, 'rb'))
[f1, f2, w3, w4, w5, b1, b2, b3, b4, b5] = params

test_data, test_label = read_data("./Coins/TestData/")
corr = 0
classes = ["Chinese", "Ottoman", "Roman"]
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(wspace=0, hspace=0.2)


for ind, img in enumerate(test_data):
    pred, prob = predict(img, f1, f2, w3, w4, w5, b1, b2, b3, b4, b5)
    if pred == test_label[ind]:
        corr += 1

    disp = Image.fromarray(img.reshape(256, 256, 3).astype(np.uint8), "RGB")
    sub = fig.add_subplot(5, 6, ind+1)
    res = int(test_label[ind] == pred)
    bgc = "g" if res else "r"
    sub.set_title(str(classes[pred]), fontsize="large",fontweight="bold",
                  color="white", backgroundcolor=bgc)
    plt.axis('off')
    plt.imshow(disp)

print("Overall Accuracy of Model: %.2f" % (float(corr / len(test_data) * 100)))
plt.show()
