import time
import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model
import soundfile

lang = 'English'
fs = 22050
tag = 'kan-bayashi/ljspeech_tacotron2'
vocoder_tag = "ljspeech_parallel_wavegan.v1"


d = ModelDownloader()
text2speech = Text2Speech(
    **d.download_and_unpack(tag),
    device="cuda",
    # Only for Tacotron 2
    threshold=0.5,
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2
    speed_control_alpha=1.0,
)
text2speech.spc2wav = None
vocoder = load_model(download_pretrained_model(vocoder_tag)).to("cuda").eval()
vocoder.remove_weight_norm()

start = time.time()
x = input()
# synthesis
total_secs = 0
with torch.no_grad():
    wav, c, *_ = text2speech(x)
    wav = vocoder.inference(c)
total_secs = round(time.time() - start)
print(total_secs)
print("Time in seconds = {}".format(total_secs))
# save the file
soundfile.write("outputs/out.wav", wav.detach().numpy(), fs, "PCM_16")