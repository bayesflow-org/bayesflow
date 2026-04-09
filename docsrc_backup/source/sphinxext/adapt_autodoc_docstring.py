def docstring(app, what, name, obj, options, lines):
    """Adapt autodoc docstring.

    Inherited Keras docstrings do not follow the numpy docstring format.
    Most noticably, they use Markdown code fences. To improve the situation
    somewhat, this functions aims to convert fenced code blocks to RST
    code blocks.
    """
    updated_lines = []
    prefix = ""
    for line in lines:
        if line.count("```") == 1:
            if prefix == "":
                prefix = "    "
                updated_lines[-1] = updated_lines[-1] + "::"
            else:
                prefix = ""
            updated_lines.append("\n")
        else:
            updated_lines.append(prefix + line)
    if prefix != "":
        raise ValueError(
            f"Uneven number of code fences in docstring:\nwhat='{what}', name='{name}'.\n" + "\n".join(lines)
        )
    # overwrite lines list
    lines.clear()
    lines += updated_lines


def setup(app):
    app.connect("autodoc-process-docstring", docstring)
