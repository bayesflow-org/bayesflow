import os

SWITCH_VERSION = "v1.1.6"

TEMPLATE_LATEST = """<!doctype html>
<html>
    <head>
    <title>Redirecting to {{{{ latest.name }}}} branch/version</title>
        <meta charset="utf-8" />
        <meta
            http-equiv="refresh"
            content="0; url=/{{{{ latest.name }}}}/{path}"
        />
        <link rel="canonical" href="/{{{{ latest.name }}}}/{path}" />
    </head>
</html>
"""

TEMPLATE_V1 = f"""<!doctype html>
<html>
    <head>
    <title>Redirecting to {SWITCH_VERSION}</title>
        <meta charset="utf-8" />
        <meta
            http-equiv="refresh"
            content="0; url=/{SWITCH_VERSION}/{{path}}"
        />
        <link rel="canonical" href="/{SWITCH_VERSION}/{{path}}" />
    </head>
</html>
"""


def generate_templates(old_doc_dir, include_dirs, latest_files, template_dir):
    os.makedirs(template_dir, exist_ok=True)

    for curdir, dirs, files in os.walk(old_doc_dir):
        if any([curdir.startswith(old_doc_dir + "/" + d) for d in include_dirs]) or curdir == old_doc_dir:
            html_files = [f for f in files if f.endswith("html")]
            if html_files:
                os.makedirs(os.path.join(template_dir, curdir), exist_ok=True)
            for file in html_files:
                file_path = os.path.join(curdir[(len(old_doc_dir) + 1) :], file)
                print("Writing", os.path.join(curdir[(len(old_doc_dir) + 1) :], file))
                with open(os.path.join(template_dir, file_path), "w") as f:
                    template = TEMPLATE_LATEST if file in latest_files else TEMPLATE_V1
                    content = template.format(path=file_path)
                    f.write(content)


if __name__ == "__main__":
    old_doc_dir = "../docs/v1.1.6"
    include_dirs = ["_examples", "_images", "_modules", "api"]
    latest_files = ["index.html", "about.html"]
    template_dir = "polyversion/templates"

    generate_templates(old_doc_dir, include_dirs, latest_files, template_dir)
