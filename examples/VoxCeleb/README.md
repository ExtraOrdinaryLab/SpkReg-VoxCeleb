# VoxCeleb

When first running the experiments on VoxCeleb dataset, you need to follow these steps:

1. Download the VoxCeleb dataset from [here](https://huggingface.co/datasets/confit/voxceleb-full).
2. Extract the downloaded dataset into a directory named `VoxCeleb1` for VoxCeleb1 and `VoxCeleb2` for VoxCeleb2. Each directory contains two subdirectories `dev` and `test`, and in those two subdirectories it should contain `wav` folder. For VoxCeleb2 dataset, since the audio files are in m4a format, you need to convert the m4a files to wav using the following command: `bash convert.sh`.
3. Symbolically linked `models` directory to `examples/VoxCeleb/EcapaTdnn` directory using the following command: `ln -s /path/to/models /path/to/examples/VoxCeleb/EcapaTdnn`.
4. Generate `train.tsv` and `test.tsv` with two columns, audio file path and the speaker ID, using the following command: `generate_audio_label_tsv.sh`.