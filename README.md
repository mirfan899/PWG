### installation
Install Parallel WavGan.
```shell
pip install espnet==0.9.7 parallel_wavegan==0.4.8
pip install espnet_model_zoo
pip install soundfile
pip install fask flask-restful
```

if you are using Cuda 11 install
```shell
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### Server
Run the flask server to test the model.
```shell
python api.py
```

It will take some time to load the model, after loading you can run the `test_api.py` to test the model by providing the
text or sentence of your choice.
```shell
python test_api.py
```

Or you can run the curl command to test the model
```shell
curl -d '{"sentence":"this is a sentence"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/api/pwg
```
### Independent Scripts.
Now run `test_gpu.py` file, it will ask for input. Provide sentences to generate the audio. It will save the audio
as `out.wav` file.

Same goes for cpu `test_cpu.py` file, it will ask for input. Provide sentences to generate the audio. It will save the audio
as `out.wav` file.


