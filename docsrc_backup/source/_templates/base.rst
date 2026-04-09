{{ objname | escape | underline(line="=") }}

{% if objtype == "function" -%}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{%- elif objtype == "class" -%}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __call__
   :member-order: bysource
   :undoc-members:

{%- else -%}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{%- endif -%}
