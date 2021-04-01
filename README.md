### installation
Install Parallel WavGan.
```shell
pip install espnet==0.9.7 parallel_wavegan==0.4.8
pip install espnet_model_zoo
pip install soundfile
```

Now run `test_gpu.py` file, it will ask for input. Provide sentences to generate the audio. It will save the audio
as `out.wav` file.

Same goes for cpu `test_cpu.py` file, it will ask for input. Provide sentences to generate the audio. It will save the audio
as `out.wav` file.

```shell
curl -d '{"sentence":"this is a sentence"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/api/wav
```
