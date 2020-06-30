# Tezos Prediction

This repository explores the use of technical indicators to predict the value
of Tezos (XTZ) over time. Though technical analysis is often frowned upon in
comparison with fundamental analysis, recent work
[[1]](https://arxiv.org/pdf/1901.05237.pdf)
[[2]](http://www.ieee-jas.org/fileZDHXBEN/journal/article/zdhxbywb/2020/3/PDF/JAS-2019-0392.pdf)
has demonstrated the effectiveness of CNN's on market prediction tasks - without
the need for extra-market data sources (e.g. sentiment analysis of headlines).

Since early 2020, I've been collecting data on Bitcoin and Tezos every 6
minutes. This includes:
- Rank (based on Market Cap)
- Market Cap
- Price
- 24 Hour Volume
- Percent Change (last hour)
- Percent Change (last week)

You can download the dataset and trained models
[here](https://drive.google.com/drive/folders/1m8Km28rn6RPEPKIJ_gAsp8aV6TJU2Sfq?usp=sharing).

## Architecture
The models I've released here look only at price over time. Specifically,
they see 40 time steps and predict the next 5. At 6 minutes per time step,
that's like knowing the past 4 hours and predicting the next half hour.

The CNN architecture receives Grammian Angular Fields as input. This is
basically a transformation from the time domain to the frequency domain
in polar coordinates (at least that's how I think of it).The LSTM, on the
other hand, see the raw time series.

For comparison purposes, both networks train for 6 epochs with batch sizes
of 16. Both use the mean squared error as their loss function, and are
similar in terms of parameter count (277k vs 159k). Obviously, it's
possible that optimizations could be made to improve the accuracy of each
model, but I'm really just doing this as an exploratory exercise.

## Findings
Model | Mean Squared Error | Mean Absolute Error
--- | :---: | :---: |
CNN | 0.275 | 0.4458
LSTM | 1.095 | 0.6432

It's clear that the CNN with GAF images outperforms the LSTM on this dataset
(at least with these hyperparameters). This result is in line with recent
research in the field.

## Prerequisites
To run this code, you will need `numpy`, `scipy`, and `tensorflow.keras`. I
recommend installing via a package manager like conda, but that's up to you.

## Usage
Download the dataset from the link above, unzip it, and place the 'dataset'
folder in the repository's root directory (on the same level as 'models',
'testing', 'training', and 'transforms').

Once you have the dataset, you can replicate my results by calling `python
main.py`, which will train (1) a CNN on GAF images and (2) an LSTM on 1D
sequences. It will then save the model weights to disk and evaluate both
architectures on the test set.

If you want to dig deeper or run your own experiments, start with the files
in the 'training' and 'testing' folders. You'll be able to modify all of the
hyperparameters. If you want to change the architecture itself, look in the
'models' folder.

## Disclaimer
I don't recommend using these models in combination with any sort of trading
bot or trading strategy. This work is for research purposes only. Use at your
own risk.
