This package gives an out of the box solution for gaze estimation.

Installation
------------

.. code-block::

    pip install gazetimation

Usage
-----

.. code-block:: python
    :linenos:

    from gazetimation import Gazetimation
    gz = Gazetimation()
    gz.run()


The solution can be customized by passing parameters to the :py:class:`Gazetimation <gazetimation.Gazetimation>` constructor, and to the :py:meth:`run <gazetimation.Gazetimation.run>` method.

Let's take a look at the :py:class:`Gazetimation <gazetimation.Gazetimation>` constructor.
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gazetimation.Gazetimation.__init__

.. important::
    | If you want to use a video file and not the direct video feed, you don't need to pass any value for `device` argument.
    | If `video_path` is provided, then the system disregards the camera feed.


Let's go through the :py:meth:`run method <gazetimation.Gazetimation.run>`.
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gazetimation.Gazetimation.run

.. attention::
    If you're not sure about how many people (faces) are present in the scene you can use the :py:meth:`find_face_num <gazetimation.Gazetimation.find_face_num>` method.



Issues
~~~~~~
If any issues are found, they can be reported `here <https://github.com/paul-shuvo/gazetimation/issues>`__.

License
~~~~~~~

This project is licensed under the `MIT <https://opensource.org/licenses/MIT>`__ license.


Acknowledgement
---------------
This package was inspired from the amazing `Medium post <https://medium.com/mlearning-ai/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23>`__ by Amit Aflalo