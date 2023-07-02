from robocat.models.vqgan.VQGAN import VQGAN_F8_8192
from robocat.models.gato.GATO.gato import GatoConfig, Gato
import torch
#
class RoboCat2:
    def __init__(self, device, vqgan_config=None, gato_config=None):
        self.device = device
        self.vqgan = VQGAN_F8_8192(self.device)
        self.gato_config = gato_config if gato_config else GatoConfig.small()
        self.gato = Gato(self.gato_config)

    def collect_and_preprocess_data(self, data_urls):
        try:
            return self.vqgan.collect_and_preprocess_data(data_urls)
        except Exception as e:
            print("Error while collecting and preprocessing data: ", e)
            return None

    def encode_images(self, preprocessed_data):
        encoded_images = []
        try:
            for img_tensor in preprocessed_data:
                z, _, [_, _, indices] = self.vqgan.model.encode(img_tensor)
                encoded_images.append(z)
            return encoded_images
        except Exception as e:
            print("Error while encoding images: ", e)
            return None

    def feed_to_gato(self, input_ids, encoding, row_pos, col_pos, obs):
        try:
            hidden_states = self.gato((input_ids, (encoding, row_pos, col_pos), obs))
            return hidden_states
        except Exception as e:
            print("Error while feeding encoded images to GATO: ", e)
            return None

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
robocat = RoboCat2(device)

data_urls = ["https://images.unsplash.com/photo-1592194996308-7b43878e84a6", 
             "https://images.unsplash.com/photo-1582719508461-905c673771fd"]

preprocessed_data = robocat.collect_and_preprocess_data(data_urls)
encoded_images = robocat.encode_images(preprocessed_data)

# The following tensor inputs for GATO need to be defined or calculated as in your original GATO setup
input_ids = torch.cat(encoded_images, dim=1)
encoding = torch.tensor([[0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2]])
row_pos = (torch.tensor([[0.00, 0.25, 0.50, 0.75, 0, 0, 0.00, 0.25, 0.50, 0.75, 0, 0]]),  
           torch.tensor([[0.25, 0.50, 0.75, 1.00, 0, 0, 0.25, 0.50, 0.75, 1.00, 0, 0]]))
col_pos = (torch.tensor([[0.00, 0.00, 0.00, 0.80, 0, 0, 0.00, 0.00, 0.00, 0.80, 0, 0]]),  
           torch.tensor([[0.20, 0.20, 0.20, 1.00, 0, 0, 0.20, 0.20, 0.20, 1.00, 0, 0]]))
obs = (torch.tensor([[ 0,  1,  2, 19, 20, 21,  0,  1,  2, 19, 20, 21]]),  
       torch.tensor([[ 1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0]]))

hidden_states = robocat.feed_to_gato(input_ids, encoding, row_pos, col_pos, obs)





