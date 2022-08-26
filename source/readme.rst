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


A lot of customization can be done by passing parameters to `Gazetimation()` and `run()`.
:py:class:`gazetimation.Gazetimation`, :py:meth:`gazetimation.Gazetimation.run`




Acknowledgement
===============
This package was inspired from the amazing `Medium post <https://medium.com/mlearning-ai/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23>` by Amit Aflalo