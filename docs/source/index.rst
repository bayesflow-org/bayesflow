:html_theme.sidebar_secondary.remove: true

#######################
Bayesflow documentation
#######################

.. toctree::
   :maxdepth: 1
   :hidden:

   Getting Started <getting_started/index>
   User Guide <user_guide/index>
   API reference <api/bayesflow>
   About us <about/index>
   Development <development/index>


BayesFlow is a Python library for efficient Bayesian inference with deep learning.
It provides users with:

.. container:: bf-feature-section

   .. grid:: 1 1 3 3
      :gutter: 2

      .. grid-item-card::
         :class-card: bf-feature-card
         :text-align: left

         Familiar API
         ^^^

         A user-friendly API for `amortized Bayesian workflows <https://arxiv.org/abs/2409.04332>`_

      .. grid-item-card::
         :class-card: bf-feature-card
         :text-align: left

         Neural architectures
         ^^^

         A rich collection of `neural network architectures <https://arxiv.org/abs/2512.20685>`_

      .. grid-item-card::
         :class-card: bf-feature-card
         :text-align: left

         Multi-backend support
         ^^^

         Multi-backend support via `Keras3 <https://keras.io/keras_3/>`_:
         You can use `PyTorch <https://github.com/pytorch/pytorch>`_,
         `TensorFlow <https://github.com/tensorflow/tensorflow>`_, or
         `JAX <https://github.com/google/jax>`_


###################
Conceptual Overview
###################
.. container:: homepage-hero

   .. grid:: 1 1 1 1
      :gutter: 3

      .. grid-item::

         .. image:: _static/img/bayesflow_landing_light.png
            :width: 100%
            :alt: BayesFlow banner
            :class: only-light

         .. image:: _static/img/bayesflow_landing_dark.png
            :width: 100%
            :alt: BayesFlow banner
            :class: only-dark


A cornerstone idea of amortized Bayesian inference is to employ generative neural
networks for parameter estimation, model comparison, and model validation when
working with intractable simulators whose behavior as a whole is too complex to
be described analytically.

.. grid:: 1 1 3 3
   :gutter: 2 3 4 4

   .. grid-item-card::
      :img-top: ../source/_static/icons/getting_started.svg
      :text-align: center

      Getting started
      ^^^

      New to Bayesflow? Start with the Absolute Beginner's Guide for the basics
      and links to tutorials.

      +++

      .. button-ref:: getting_started/index
         :expand:
         :color: secondary
         :click-parent:

         To the getting started

   .. grid-item-card::
      :img-top: ../source/_static/icons/user_guide.svg
      :text-align: center

      User guide
      ^^^

      The user guide explains Bayesflow’s key concepts with helpful background details.

      +++

      .. button-ref:: user_guide/index
         :expand:
         :color: secondary
         :click-parent:

         To the user guide

   .. grid-item-card::
      :img-top: ../source/_static/icons/api.svg
      :text-align: center

      API reference
      ^^^

      The reference guide details Bayesflow’s functions, modules, and objects, including how they work and available parameters. It assumes familiarity with the core concepts.

      +++

      .. button-ref:: api/bayesflow
         :expand:
         :color: secondary
         :click-parent:

         To the reference guide