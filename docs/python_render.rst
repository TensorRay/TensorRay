.. _Python render:


Rendering a scene
====================

TensorRay renders virtual scenes described using a simplified version of Mitsuba's XML `description language <https://mitsuba2.readthedocs.io/en/latest/src/getting_started/file_format.html#sec-file-format>`_.

To set up CUDA and OptiX, the main computation of a Python script should be surrounded as follows:

.. code-block:: python3

   import TensorRay as TR
   
   TR.env_create()

   # Main computation
   # ...

   TR.env_release()


Loading a scene
--------------------

TensorRay provides this function ``TR.Scene.load_file()`` for loading a scene from an XML file on disk.

Here is a Python example on how to load a scene from an XML file:

.. code-block:: python3

   import TensorRay as TR

   scene = TR.Scene()
   scene.load_file('scene.xml')
   scene.configure()

Note that a scene must be configured before being rendered. If the scene is loaded without being manually configured via ``scene.configure()``, attempting to render it will cause a runtime error.


Forward rendering of a scene
------------------------------

Before rendering, we need to set sample counts per pixel using ``RenderOptions``:

.. code-block:: python3

   options = TR.RenderOptions(
      seed,                # Random seed
      max_bounce,          # Maximum number of bounces for rays
      spp,                 # Spp for the main image/interior integral
      sppe,                # Spp for primary edge integral
      sppse0,              # Spp for direct secondary edge integral
      sppse1,              # Spp for indirect secondary edge integral
      sppe0                # Spp for pixel edge integral (required for the box pixel filter)
   )

To better utilize the GPU and reduce the rendering time, we recommend using larger batch sizes (set to 1 by default) for spp as long as the GPU memory fits. For example, we can set batch sizes as follows:

.. code-block:: python3

   options.spp_batch = 16
   options.sppe_batch = 64
   options.sppse0_batch = 32
   options.sppse1_batch = 32
   options.sppe0_batch = 64


Then, we can render the scene as follows:

.. code-block:: python3

   # Initialize an integrator
   integrator = TR.PathTracer()

   # Start rendering!
   image = integrator.renderC(scene, options)


The ``image`` variable returned by the integrator's ``renderC()`` function is an RGB image stored as a TensorRay tensor of the type ``TR.Tensorf`` (the order of its dimensions is [3, height, width]). This variable can be converted to a Numpy array and saved as an OpenEXR image named ``image.exr`` using OpenCV:

.. code-block:: python3

   from pyTensorRay.utils import image_tensor_to_torch, save_torch_image

   height = scene.get_height(0)
   width = scene.get_width(0)
   image = image_tensor_to_torch(image, height, width)
   save_torch_image("image.exr", image)
