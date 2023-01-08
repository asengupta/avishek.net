---
title: "A Pipeline for Adaptive Bitrate Video Encoding"
author: avishek
usemathjax: true
tags: ["Video Processing", "Archive", "Software Engineering"]
draft: false 
---

**Note:** This is a post from July 13, 2011, rescued from my old blog. This is only for archival purposes, and is reproduced verbatim, but is hopelessly outdated.

I’ve been working on something unusual lately, namely, building a pipeline for encoding video files into formats suitable for **HTTP Live Streaming**. The actual job of encoding into different formats at different bit rates and resolutions is done using a combination of ffmpeg and x264. To me, the interesting part lies in how we have tried to speed up the process, using the venerable Map-Reduce approach. Before I dive into the details, here’s a quick review of the basic idea of HLS.

Put very simply, adaptive streaming serves video content in multiple qualities, allowing the streaming client choice in selecting which quality to use depending upon the bandwidth constraint on the consumer side. This choice is not a one-time choice, depending upon the encode cut duration, the client can switch to higher or lower resolutions dynamically throughout the entire playback of the video stream.
How is this accomplished?

Assume that you slice up a video into mulitple segments. Each of these segments can be as long or as short as you want; for argument’s sake, I shall assume that every segment lasts 10 seconds. Now, encode each of these segments at different levels of quality, say a low bit rate for mobile consumption, a high quality one for fat broadband connections, etc. What you ultimately get is many versions of each segment of video, each version of a different quality.
What you ultimately publish on your server is essentially a list of the names of these segments. This is the playlist for a video encoded with adaptive streaming. When the client asks you to open a network stream, the URL you put in is that of the playlist. The situation then looks something like this.

![Figure 1](/assets/images/adaptive-video-encoding-figure-1.png)

The client ends up retrieving a master playlist which links to the quality-specific playlists that you see in the picture above. That’s more or less the general idea.
One thing to note is that even though the stream is served in segments, the segments themselves do not have to necessarily be separate files. In fact, Microsoft’s take on adaptive streaming involves internally fragmenting a video file to generate a single .ismv file per quality. This .ismv file is internally fragmented, so even though it’s a single file in the filesystem, its content is served in segments.
That is how we’ve chosen to serve these files. CodeShop has a tool called mp4split which generates these .ismv files. It can also generate playlists in different formats; for example, Apple’s .m3u8 format, Silverlight’s .ismc format and Flash’s .f4v format.
Of course, simply having these files isn’t enough, since the the segments ‘reside’ inside the single .ismv file. The server needs to recognise requests for segments and extract those segments.
For this purpose, we’ve used CodeShop’s Nginx module for serving these files; see here and here.

So we wanted to implement this. The only issue is that for a large video file, encoding takes a very long time. Encoding is a parallelisable task; you can partition a file into small segments and encode them independently. This is how our pipeline looks like:

![Figure 2](/assets/images/adaptive-video-encoding-figure-2.png)

**Transcode Stage:** This stage essentially transcodes the video from its original format (.avi/.mov, etc.) into an MP4 format using H264 and AAC encodings for video and audio, respectively. This is done using the x264 utility.

**Split Stage:** This splits the MP4 video into 2-second segments, and generates information which will be used by the adaptive streaming encoding processes to determine which quality to encode to.

**Encode Stage:**This is where the parallelism comes in. Basically, multiple daemons pick up assignments from a (Starling) queue, and encode the segments at the desired qualities, this is one daemon per segment per encoding level (quality).

**Merge Stage:** This is where the encoded segments are joined back to form a single .mp4 video file per encoding level.

**Fragment and Generate Playlist:** The video files are internally fragmented using mp4split, and the playlist files are generated.
