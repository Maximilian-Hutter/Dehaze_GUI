from tkinter import *
from PIL import ImageTk, Image
from models import Dehaze
import torchvision.transforms as T
import torch
from params import hparams

class Generator():
    def __init__(self, path) -> None:

        model=Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"], hparams["down_deep"], pseudo_alpha= hparams["pseudo_alpha"], hazy_alpha=hparams["hazy_alpha"])
        model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        model = model.eval()
        model = model.cpu()
        self.model = model

    def getModel(self):
        return self.model

def Generate(path):
    image = Image.open("curr.png")
    transformtotensor = T.Compose([T.ToTensor()])
    image = transformtotensor(image)
    image = image.unsqueeze(0)
    image= image.to(torch.float32)
    image = image.to(torch.device('cpu'))

    generator = Generator(path)
    model = generator.getModel()
    image,_ = model(image)
    transform = T.ToPILImage()
    image = transform(image.squeeze(0))

    output = ImageTk.PhotoImage(image)
    outputlabel = Label(image=output).grid(row=0, column=40,padx=10)
    outputlabel.image = output


def chooseimg(file):
    image = Image.open(file)
    image.save("curr.png")

window=Tk()

img1_file = Image.open("./0.png")
img1_file = img1_file.resize([320,180])
img1 = ImageTk.PhotoImage(img1_file)
img_label = Label(image=img1)
button= Button(window, image=img1,command=lambda: chooseimg("0.png"),borderwidth=0).grid(row=0, column=0,padx=10)


img2_file = Image.open("./1.png")
img2_file = img2_file.resize([320,180])
img2 = ImageTk.PhotoImage(img2_file)
img_label2 = Label(image=img2)
button2= Button(window, image=img2,command=lambda: chooseimg("1.png"),borderwidth=0).grid(row=20, column=0,padx=10)


img3_file = Image.open("./2.png")
img3_file = img3_file.resize([320,180])
img3 = ImageTk.PhotoImage(img3_file)
img_label3 = Label(image=img3)
button3= Button(window, image=img3,command=lambda: chooseimg("2.png"),borderwidth=0).grid(row=40, column=0,padx=10)


btn=Button(window, text=" 329 Epochs Generate", fg='black', command=lambda: Generate("329.pth")).grid(row=19, column=2,padx=50, pady=10)
btn2=Button(window, text=" 199 Epochs Generate", fg='black', command=lambda: Generate("199.pth")).grid(row=20, column=2,padx=50, pady=10)
btn3=Button(window, text=" 99 Epochs Generate", fg='black', command=lambda: Generate("99.pth")).grid(row=21, column=2,padx=50, pady=10)


window.title('Dehaze')
#window.geometry("1024x740")
window.mainloop()