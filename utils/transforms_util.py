from torchvision import transforms
import base64
from io import BytesIO


def trans_pil_to_tensor(pil_img):
    tensor_image = transforms.ToTensor()(pil_img).unsqueeze_(0)
    return tensor_image


def trans_tensor_to_pil(tensor_img):
    pil_image = transforms.ToPILImage()(tensor_img.squeeze_(0))
    return pil_image


def trans_tensor_to_pil1(tensor_img):
    pil_image = transforms.ToPILImage(tensor_img)
    return pil_image


def trans_pil_to_tensor1(pil_img):
    tensor_image = transforms.ToTensor()(pil_img)
    return tensor_image


def trans_tensor_to_b64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/;base64," + base64.b64encode(img_byte).decode()
    return img_str
