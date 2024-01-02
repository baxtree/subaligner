## Subaligner Helm Chart

This chart needs to be customised with the following parameters:

- **videoDirectory:** The path to the video directory on the host node.
- **subtitleDirectory:** The path to the subtitle directory on the host node.
- **outputDirectory:** The path to the output directory on the host node.


### Installation
```
helm install subaligner . --set videoDirectory=$VIDEO_DIRECTORY,subtitleDirectory=$SUBTITLE_DIRECTORY,outputDirectory=$OUTPUT_DIRECTORY
```