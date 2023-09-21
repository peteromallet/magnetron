# Magnetron Data Collector

Magnetron is a tool to collect precisely-tagged motion data for training video models, including [Steerable Motion](https://github.com/banodoco/steerable-motion).

To do this, we collect key pairs from videos and tag them using an automated process, and review these tags with humans. These pairs can be used to train, or you can extract the full video between them from the source.

You can see the current output from this tool [here](https://magnetron-output.streamlit.app/)

### Run this locally:

To install dependencies, run:

```
conda env create -f environment.yaml

conda activate magnetron

```

To run the tool, run:

```

streamlit run magnetron.py

```

### Collaborate on this project

We're looking for talented ML engineers, artists, fine-tuners, and hackers to collaborate on this and other projects. If you're interested, you can join our [Discord](https://discord.gg/KRVwb83hq7) here.