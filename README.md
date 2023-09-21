# Magnetron Data Collector

Magnetron is a tool to collect precisely-tagged motion data for training video models, including [Steerable Motion](https://github.com/banodoco/steerable-motion).

It's designed to collect key pairs from videos and tag them using an automated process - with human review at the end. These pairs can be used to train models, or you can extract the full video between them from the source: [Webvid](https://maxbain.com/webvid-dataset/).

You can see the current output from this tool [here](https://magnetron-output.streamlit.app/) and some examples below:

![Data Examples](https://banodoco.s3.amazonaws.com/data_examples.webp)

### To run this locally:

Currently, we use API providers for convenience. To insert relevant API keys, create a .env file in the root directory in the following format with the relevant keys you need:

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

Then, to run the tool, run:

```
streamlit run magnetron.py
```

This also includes a Discord bot for the review process. To run this, you'll need to set up a Discord bot and add it to your server. You can find instructions for this [here](https://discord.com/developers/applications/), authenticate it with your server, and then invite it to your server. You'll also need to have added the Discord bot token to your .env file - see above.

Then, to run the bot, run:

```
python review_bot.py
```

### Collaborate on this project

We're looking for talented ML engineers, artists, fine-tuners, and hackers to collaborate on this and other projects. If you're interested, you can join our [Discord](https://discord.gg/KRVwb83hq7) here.