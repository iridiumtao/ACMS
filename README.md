# Adaptive Music Controlling System
Adaptive Music Controlling System is a system that automatically plays suitable background music according to one restaurant environment picture.

We use Azure Computer Vision to obtain image tags, then use k-NN to group the tags to identify restaurant types, and then play appropriate music through Spotify Web API.

## Techs

- Python
- Data analysis
- Data pre-processing
- k-NN Algorithm
- Spotify Web API
- Microsoft Azure

## Sample Result

![Untitledhk](https://github.com/iridiumtao/ACMS/assets/43561001/d330a280-4164-499c-9371-5e4b4fe99c4e)

## Reference and APIs

Microsoft Azure

[Quickstart: Computer Vision client library for Python](https://docs.microsoft.com/azure/cognitive-services/computer-vision/quickstarts-sdk/python-sdk)

```bash
pip install azure-cognitiveservices-vision-computervision
```

[Spotipy](https://spotipy.readthedocs.io)

```bash
pip install spotipy
```
