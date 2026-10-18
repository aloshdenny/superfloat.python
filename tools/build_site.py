"""Generate the Mintlify docs site (website/) from the Markdown in docs/, NOTES.md,
README.md and examples/.  Run after editing any of those:

    python tools/build_site.py && (cd website && mint validate)

The Markdown files stay the source of truth; website/ is generated and committed
so Mintlify can deploy it from the repository.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "website"


def mdx_escape(md: str) -> str:
    """Escape the characters MDX treats as JSX outside code spans and blocks."""
    out = []
    in_block = False
    for line in md.split("\n"):
        if line.strip().startswith("```"):
            in_block = not in_block
            out.append(line)
            continue
        if in_block:
            out.append(line)
            continue
        parts = re.split(r"(`[^`]*`)", line)
        in_table = line.lstrip().startswith("|")
        for i, p in enumerate(parts):
            if i % 2 == 1:
                if in_table:                               # a pipe inside a code span would split the cell
                    parts[i] = p.replace("|", "\\|")
                continue                                   # inside a code span
            p = p.replace("{", "\\{").replace("}", "\\}")
            p = re.sub(r"<(?=[A-Za-z/=])", "&lt;", p)      # a tag-looking "<"
            p = p.replace("<=", "&lt;=").replace(" < ", " &lt; ")
            parts[i] = p
        out.append("".join(parts))
    return "\n".join(out)


def split_sections(md: str, level: str = "## "):
    """Yield (heading, body) for each top-level section; body keeps sub-headings."""
    head, cur, buf = None, None, []
    for line in md.split("\n"):
        if line.startswith(level) and not line.startswith(level + "#"):
            if cur is not None:
                yield cur, "\n".join(buf).strip()
            elif buf:
                head = "\n".join(buf).strip()
            cur, buf = line[len(level):].strip(), []
        else:
            buf.append(line)
    if cur is not None:
        yield cur, "\n".join(buf).strip()


def page(path: str, title: str, description: str, body: str, *, promote: bool = True) -> str:
    if promote:                                            # ### inside a page become ##
        body = re.sub(r"^### ", "## ", body, flags=re.M)
    body = re.sub(r"\]\(docs/([A-Za-z]+)\.md\)", lambda m: f"]({_link(m.group(1))})", body)
    body = re.sub(r"\]\(NOTES\.md\)", "](/decisions/overview)", body)
    fm = f'---\ntitle: "{title}"\ndescription: "{description}"\n---\n\n'
    text = fm + mdx_escape(body) + "\n"
    (OUT / f"{path}.mdx").parent.mkdir(parents=True, exist_ok=True)
    (OUT / f"{path}.mdx").write_text(text)
    return path


def _link(name: str) -> str:
    return {"API": "/reference/semantics", "TESTING": "/testing/matrix", "DETAILS": "/details/semantics"}[name]


def slug(s: str) -> str:
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s


def main() -> None:
    if OUT.exists():
        for p in OUT.iterdir():
            if p.name not in ("docs.json", "logo", "favicon.svg"):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    OUT.mkdir(exist_ok=True)
    nav = {"tabs": []}

    # --- Guides: README split into pages, plus the tours ------------------------------
    readme = (ROOT / "README.md").read_text()
    intro = readme.split("## What is Superfloat")[0].split("\n", 1)[1].strip()
    sections = dict(split_sections(readme))
    guides = []
    guides.append(page("index", "Superfloat in Python", "What sf is and what it is for",
                       intro + "\n\n## What is Superfloat, in one paragraph\n\n" + sections["What is Superfloat, in one paragraph"]))
    guides.append(page("quickstart", "Install and run", "Install sf, run the tests, run the tours",
                       sections["Install and run"]))
    guides.append(page("guides/scalars", "Scalars", "Stage 1.0: values, arithmetic, comparisons",
                       _bullet_section(sections["What you can do with it"], "1.")))
    guides.append(page("guides/arrays", "Arrays and the device", "Stage 1.1: arrays, matmul, the pre-flight gate",
                       _bullet_section(sections["What you can do with it"], "2.")))
    guides.append(page("guides/pytorch", "PyTorch models", "Stage 2: quantize, convert, compile",
                       _bullet_section(sections["What you can do with it"], "3.")))
    guides.append(page("guides/kernels", "Writing kernels", "Stage 3: thread and block kernels on the simulator",
                       _bullet_section(sections["What you can do with it"], "4.")))
    guides.append(page("guides/errors", "Errors and the policy", "What happens when a number does not fit",
                       sections["Errors: what happens when a number does not fit"]))
    tours = []
    for f in sorted((ROOT / "examples").glob("stage_*_tour.py")):
        name = f.stem.replace("_tour", "").replace("stage_", "Stage ").replace("_", ".")
        src = f.read_text()
        tours.append(page(f"tours/{f.stem}", f"{name} tour", f"examples/{f.name}, runnable end to end",
                          f"Run with `python examples/{f.name}`.\n\n```python\n{src}\n```", promote=False))
    guides.append(page("status", "What is done and what is not", "Stage by stage status and the open hardware work",
                       sections["What is done and what is not"]))
    nav["tabs"].append({"tab": "Guides", "icon": "book-open", "groups": [
        {"group": "Start here", "pages": guides[:2]},
        {"group": "Using sf", "pages": guides[2:7]},
        {"group": "Tours", "pages": tours},
        {"group": "Project", "pages": guides[7:]},
    ]})

    # --- Reference: API.md, one page per numbered section ----------------------------------
    api = (ROOT / "docs" / "API.md").read_text()
    ref_pages = []
    names = {"python-semantics": ("semantics", "Python semantics"),
             "stage-1-0-scalars": ("scalars", "Scalars (Stage 1.0)"),
             "stage-1-1-arrays-reductions-device-path": ("arrays", "Arrays and the device path (Stage 1.1)"),
             "error-policy-and-configuration": ("config", "Error policy and configuration"),
             "stage-2-sf-torch": ("torch", "sf.torch (Stage 2)"),
             "stage-3-kernels": ("kernels", "Kernels (Stage 3)"),
             "exceptions-and-warnings": ("errors", "Exceptions and warnings"),
             "objects": ("objects", "Objects"),
             "plugging-in-a-real-simulator-or-the-hardware": ("hardware", "Plugging in a simulator or the hardware")}
    for heading, body in split_sections(api):
        key = slug(heading)
        if key not in names:
            continue
        path, title = names[key]
        body = body.replace("\n---\n", "\n")
        ref_pages.append(page(f"reference/{path}", title, heading, body))
    nav["tabs"].append({"tab": "API reference", "icon": "code", "groups": [
        {"group": "Semantics", "pages": ref_pages[:1]},
        {"group": "Stages 1 to 3", "pages": ref_pages[1:6]},
        {"group": "Types", "pages": ref_pages[6:8]},
        {"group": "Hardware", "pages": ref_pages[8:]},
    ]})

    # --- Testing ---------------------------------------------------------------------------------
    testing = (ROOT / "docs" / "TESTING.md").read_text()
    t_intro, *t_rest = re.split(r"^## ", testing, flags=re.M)
    t_pages = [page("testing/matrix", "Testing matrix", "What is checked, against what, at which widths, when",
                    t_intro.split("\n", 1)[1].strip())]
    for sec in t_rest:
        heading, body = sec.split("\n", 1)
        t_pages.append(page(f"testing/{slug(heading)}", heading.strip(), f"Testing: {heading.strip().lower()}", body.strip()))
    nav["tabs"].append({"tab": "Testing", "icon": "flask", "groups": [{"group": "Testing", "pages": t_pages}]})

    # --- Details and decisions ------------------------------------------------------------------
    details = (ROOT / "docs" / "DETAILS.md").read_text()
    d_pages = []
    for heading, body in split_sections(details):
        if heading.startswith("Testing") or heading.startswith("Tests"):
            continue
        d_pages.append(page(f"details/{slug(heading)}", heading, f"Detailed reference: {heading.lower()}", body))
    notes = (ROOT / "NOTES.md").read_text()
    n_pages = [page("decisions/overview", "Design notes", "How to read the decisions log",
                    notes.split("\n## ", 1)[0].split("\n", 1)[1].strip() or "Decisions taken while building sf, by stage.")]
    for heading, body in split_sections(notes):
        n_pages.append(page(f"decisions/{slug(heading)}", heading, f"Decision log: {heading.lower()}", body))
    nav["tabs"].append({"tab": "Internals", "icon": "gear", "groups": [
        {"group": "Detailed reference", "pages": d_pages},
        {"group": "Decisions", "pages": n_pages},
    ]})

    docs_json = {
        "$schema": "https://mintlify.com/docs.json",
        "theme": "mint",
        "name": "Superfloat Python",
        "colors": {"primary": "#0D9373", "light": "#55D799", "dark": "#0D9373"},
        "favicon": "/favicon.svg",
        "navigation": nav,
        "navbar": {"links": [{"label": "GitHub", "href": "https://github.com/aloshdenny/superfloat.python"}]},
        "footer": {"socials": {"github": "https://github.com/aloshdenny/superfloat.python"}},
    }
    (OUT / "docs.json").write_text(json.dumps(docs_json, indent=2) + "\n")
    (OUT / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#0D9373"/>'
        '<text x="32" y="42" font-family="monospace" font-size="28" text-anchor="middle" fill="#fff">sf</text></svg>\n')
    n = sum(1 for _ in OUT.rglob("*.mdx"))
    print(f"wrote {n} pages to {OUT.relative_to(ROOT)}/")


def _bullet_section(body: str, number: str) -> str:
    """One of the '**N. Title**' blocks of the README's 'What you can do' section."""
    blocks = re.split(r"^\*\*(?=\d\.)", body, flags=re.M)
    for b in blocks:
        if b.startswith(number):
            title, rest = b.split("**", 1)
            return rest.strip()
    raise KeyError(number)


if __name__ == "__main__":
    main()
