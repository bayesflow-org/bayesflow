# Adapted from https://github.com/scikit-learn/scikit-learn
# doc/sphinxext/override_pst_pagetoc.py, which is under the following license:
#
# BSD 3-Clause License
#
# Copyright (c) 2007-2024 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from functools import cache

from sphinx.util.logging import getLogger

logger = getLogger(__name__)


def override_pst_pagetoc(app, pagename, templatename, context, doctree):
    """Overrides the `generate_toc_html` function of pydata-sphinx-theme for API."""

    @cache
    def generate_api_toc_html(kind="html"):
        """Generate the in-page toc for an API page.

        This relies on the `generate_toc_html` function added by pydata-sphinx-theme
        into the context. We save the original function into `pst_generate_toc_html`
        and override `generate_toc_html` with this function for generated API pages.

        The pagetoc of an API page would look like the following:

        <ul class="visible ...">               <-- Unwrap
         <li class="toc-h1 ...">               <-- Unwrap
          <a class="..." href="#">{{obj}}</a>  <-- Decompose

          <ul class="visible ...">
           <li class="toc-h2 ...">
            ...object
            <ul class="...">                          <-- Set visible if exists
             <li class="toc-h3 ...">...method 1</li>  <-- Shorten
             <li class="toc-h3 ...">...method 2</li>  <-- Shorten
             ...more methods                          <-- Shorten
            </ul>
           </li>
           <li class="toc-h2 ...">...gallery examples</li>
          </ul>

         </li>                                 <-- Unwrapped
        </ul>                                  <-- Unwrapped
        """
        soup = context["pst_generate_toc_html"](kind="soup")

        try:
            # Unwrap the outermost level
            soup.ul.unwrap()
            soup.li.unwrap()
            soup.a.decompose()

            # Get all toc-h2 level entries, where the first one should be the function
            # or class, and the second one, if exists, should be the examples; there
            # should be no more than two entries at this level for generated API pages
            lis = soup.ul.select("li.toc-h2")
            main_li = lis[0]
            meth_list = main_li.ul

            if meth_list is not None:
                # This is a class API page, we remove the class name from the method
                # names to make them better fit into the secondary sidebar; also we
                # make the toc-h3 level entries always visible to more easily navigate
                # through the methods
                meth_list["class"].append("visible")
                for meth in meth_list.find_all("li", {"class": "toc-h3"}):
                    target = meth.a.code.span
                    target.string = target.string.split(".", 1)[1]

            # This corresponds to the behavior of `generate_toc_html`
            return str(soup) if kind == "html" else soup

        except Exception as e:
            # Upon any failure we return the original pagetoc
            logger.warning(f"Failed to generate API pagetoc for {pagename}: {e}; falling back")
            return context["pst_generate_toc_html"](kind=kind)

    # Override the pydata-sphinx-theme implementation for generate API pages
    if pagename.startswith("api/"):
        context["pst_generate_toc_html"] = context["generate_toc_html"]
        context["generate_toc_html"] = generate_api_toc_html


def setup(app):
    # Need to be triggered after `pydata_sphinx_theme.toctree.add_toctree_functions`,
    # and since default priority is 500 we set 900 for safety
    app.connect("html-page-context", override_pst_pagetoc, priority=900)
