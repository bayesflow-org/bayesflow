{% if objname == 'bayesflow' %}
API Reference
=============

This is the reference for the public API. For a more guided introduction,
please refer to the :doc:`Examples <../examples>` section.

{% else %}
{{ objname | escape | underline}}
{% endif %}

.. automodule:: {{ fullname }}
  :member-order: alphabetical

  {% block modules %}
  {% if modules %}
  .. rubric:: Modules

  .. autosummary::
    :toctree:
    :template: custom-module-template.rst
    :recursive:
  {% for item in modules %}
    {{ item }}
  {%- endfor %}

  {% endif %}
  {% endblock %}

  {% block attributes %}
  {% if attributes %}
  .. rubric:: Module Attributes

  .. autosummary::
  {% for item in attributes %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}

  {% block functions %}
  {% if functions %}
  .. rubric:: {{ _('Functions') }}

  .. autosummary::
    :toctree:
    :template: base.rst
  {% for item in functions %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}

  {% block classes %}
  {% if classes %}
  .. rubric:: {{ _('Classes') }}

  .. autosummary::
    :toctree:
    :template: base.rst
  {% for item in classes %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}

  {% block exceptions %}
  {% if exceptions %}
  .. rubric:: {{ _('Exceptions') }}

  .. autosummary::
    :toctree:
    :template: base.rst
  {% for item in exceptions %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}

 {% block attributes_def %}
  {% if attributes %}
  .. rubric:: Module Attributes

  .. autoattribute::
  {% for item in attributes %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}
