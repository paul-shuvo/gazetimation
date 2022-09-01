.. |github| image:: https://avatars.githubusercontent.com/in/15368?s=64&v=4
    :target: <https://github.com/paul-shuvo/gazetimation
    :width: 17px

**Gazetimation** provides an out of the box solution for gaze estimation. |github|

.. image:: https://media4.giphy.com/media/7B7Hhz6w2TCBQikqoL/giphy.gif?cid=790b76112cc0a01f4cc4de64efea7cf5f8c9c0f898ceb1a0&rid=giphy.gif&ct=g
   :width: 600px
   :align: center

Installation
------------

.. code-block::

    pip install gazetimation

Usage
-----

.. code-block:: python
    :linenos:

    from gazetimation import Gazetimation
    gz = Gazetimation(device=0) # or any other device id
    gz.run()

To run a video file

.. code-block:: python

    gz.run(video_path='path/to/video')

To save as a video file

.. code-block:: python

    gz.run(video_output_path='path/to/video.avi')

The :py:meth:`run <gazetimation.Gazetimation.run>` method also accepts a handler function for further processing.

.. code-block:: python
    
    gz.run(handler=my_handler)

.. attention::
    The handler function will be called by passing the frame and the gaze information

    .. code-block:: python
    
        if handler is not None:
            handler([frame, left_pupil, right_pupil, gaze_left_eye, gaze_right_eye])

.. For more info check our `docs <https://paul-shuvo.github.io/gazetimation/>`__.

The solution can be customized by passing parameters to the :py:class:`Gazetimation <gazetimation.Gazetimation>` constructor, and to the :py:meth:`run <gazetimation.Gazetimation.run>` method.

Let's take a look at the :py:class:`Gazetimation <gazetimation.Gazetimation>` constructor.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gazetimation.Gazetimation.__init__
    :noindex:

.. important::
    | If you want to use a video file and not the direct video feed, you don't need to pass any value for `device` argument.
    | If `video_path` is provided, then the system disregards the camera feed.


Let's go through the :py:meth:`run method <gazetimation.Gazetimation.run>`.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gazetimation.Gazetimation.run
    :noindex:

.. attention::
    If you're not sure about how many people (faces) are present in the scene you can use the :py:meth:`find_face_num <gazetimation.Gazetimation.find_face_num>` method.



Issues
~~~~~~
If any issues are found, they can be reported `here <https://github.com/paul-shuvo/gazetimation/issues>`__.

License
~~~~~~~

This project is licensed under the `MIT <https://opensource.org/licenses/MIT>`__ license.


.. Acknowledgement
.. ~~~~~~~~~~~~~~~
.. This package was inspired from the amazing `Medium post <https://medium.com/mlearning-ai/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23>`__ by Amit Aflalo