import soundfile
import uuid
from flask import Flask
from flask_restful import Resource, Api, reqparse
import time
import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model

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


def generate_wav(sentence):
    start = time.time()
    # synthesis
    total_secs = 0
    with torch.no_grad():
        wav, c, *_ = text2speech(sentence)
        wav = vocoder.inference(c)
    total_secs = round(time.time() - start)
    print(total_secs)
    print("Time in seconds = {}".format(total_secs))
    # save the file with unique uuid
    uid = str(uuid.uuid1())[:8]
    soundfile.write("outputs/{}.wav".format(), wav.detach().numpy(), fs, "PCM_16")
    return {"sentence": sentence, "uuid": uid}


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("sentence", required=True, type=str, help="please provide the sentence to get perplexity score.")


class ParallelWavGAN(Resource):
    def get(self):
        return {"message": "Welcome to Transformers Perplexity API", "status": 200}

    def post(self):
        args = parser.parse_args()
        if args["sentence"]:
            result = generate_wav(args["sentence"])
            return result
        else:
            return {"You should provide a sentence."}


api.add_resource(ParallelWavGAN, '/api/pwg')

if __name__ == '__main__':
    app.run(debug=False, threaded=False, processes=1)
