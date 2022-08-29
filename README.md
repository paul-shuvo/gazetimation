<div align="center">

<img src="/docs/source/assets/gazetimation_logo.png" />

# Gazetimation

![test](https://github.com/paul-shuvo/gazetimation/actions/workflows/test.yml/badge.svg) [![Docs](https://github.com/paul-shuvo/gazetimation/actions/workflows/docs.yml/badge.svg)](https://paul-shuvo.github.io/gazetimation/) [![PyPI version](https://badge.fury.io/py/gazetimation.svg)](https://badge.fury.io/py/gazetimation) [![License: MIT](https://img.shields.io/github/license/paul-shuvo/gazetimation)](https://opensource.org/licenses/MIT) ![downloads](https://img.shields.io/pypi/dm/gazetimation?color=blue) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/1822d5b3047a4e3596404b4c0e636912)](https://www.codacy.com/gh/paul-shuvo/gazetimation/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=paul-shuvo/gazetimation&amp;utm_campaign=Badge_Grade)

<p>An out of the box solution for gaze estimation.

<img src="https://media4.giphy.com/media/7B7Hhz6w2TCBQikqoL/giphy.gif?cid=790b76112cc0a01f4cc4de64efea7cf5f8c9c0f898ceb1a0&rid=giphy.gif&ct=g" alt="demo" width="600"/>

<!-- ![](docs/source/assets/demo.gif) -->

</div>

## Installation

```console
pip install gazetimation
```

## Usage

```python
from gazetimation import Gazetimation
gz = Gazetimation(device=0) # or any other device id
gz.run()
```

To run a video file
```python
gz.run(video_path='path/to/video')
```

To save as a video file
```python
gz.run(video_output_path='path/to/video.avi')
```

The [`run`](https://paul-shuvo.github.io/gazetimation/gazetimation.html#gazetimation.Gazetimation.run) method also accepts a handler function for further processing.
```python
gz.run(handler=my_handler)
```
__Attention__

The handler function will be called by passing the frame and the gaze information

```python
if handler is not None:
    handler([frame, left_pupil, right_pupil, gaze_left_eye, gaze_right_eye])
```

For more info check our [docs](https://paul-shuvo.github.io/gazetimation/)

## Issues

If any issues are found, they can be reported
[here](https://github.com/paul-shuvo/gazetimation/issues).

## License

This project is licensed under the
[MIT](https://opensource.org/licenses/MIT) license.

### Acknowledgement

This package was inspired from the amazing [Medium
post](https://medium.com/mlearning-ai/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23)
by Amit Aflalo
