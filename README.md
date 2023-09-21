# Magnetron Data Collector

Magnetron is a tool to collect precisely-tagged motion data for training video models, including [Steerable Motion](https://github.com/banodoco/steerable-motion).

To do this, we collect key pairs from videos and tag them using an automated process, and review them with humans. These pairs can be used to train, or you can extract the full video between them from the source.

You can see the current output from this tool [here](https://magnetron-output.streamlit.app/) and some examples below:



### Run this locally:


To insert relevant API keys, create a .env file in the root directory in the following format with the relevant keys you need:

```
DISCORD_TOKEN=
AWS_SECRET_ACCESS_KEY=
AWS_ACCESS_KEY_ID=
OPENAI_API_KEY=
REPLICATE_API_TOKEN=


```

To install dependencies, run:

```
conda env create -f environment.yml

conda activate magnetron

```

To run the tool, run:

```

streamlit run magnetron.py

```

This also includes a Discord bot for the review process. To run this, you'll need to set up a Discord bot and add it to your server. You can find instructions for this [here](https://discordpy.readthedocs.io/en/stable/discord.html).

```

streamlit run magnetron.py

```

### Collaborate on this project

We're looking for talented ML engineers, artists, fine-tuners, and hackers to collaborate on this and other projects. If you're interested, you can join our [Discord](https://discord.gg/KRVwb83hq7) here.