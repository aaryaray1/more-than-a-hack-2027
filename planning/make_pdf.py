import argparse
import json
import os
import re
import subprocess
import tempfile

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)
TEAM_ZH = "中華民國海軍"
TEAM_EN = "Republic of China Navy"
TEAM_SHORT = "ROCN"
SEAL = "技術報告"
SEAL_EN = "technical report"
KAI = '"KaiTi", "DFKai-SB", "Noto Serif SC", serif'
SERIF = 'Georgia, "Noto Serif SC", serif'
PURPLE, DEEP, RED, INK, LILAC = "#4A2A7A", "#24163F", "#C8302B", "#16181d", "#D5C6F2"
STEPS = ("tracking", "dead / alive model", "Viterbi decoder", "team model", "Viterbi + 2 s rule",
         "H / A / D per frame")
CSS = """
@page { size: A4; margin: 21mm 19mm 20mm;
  @bottom-left { content: "FOOTER"; font: 8pt SERIF; color: PURPLE; }
  @bottom-right { content: counter(page); font: 9pt SERIF; color: PURPLE; } }
@page :first { margin: 0; @bottom-left { content: none; } @bottom-right { content: none; } }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font: 10.2pt/1.6 SERIF; color: INK; }
.cover { position: relative; width: 210mm; height: 297mm; background: DEEP; color: #F5F2FA;
  padding: 30mm 22mm 26mm 26mm; display: grid; grid-template-columns: 1fr 26mm; column-gap: 12mm;
  break-after: page; }
.cover::before { content: ""; position: absolute; inset: 10mm; border: .8pt solid LILAC; opacity: .55; }
.cover::after { content: ""; position: absolute; left: 10mm; top: 10mm; bottom: 10mm; width: 3.2mm;
  background: RED; }
.cover .lead { display: flex; flex-direction: column; z-index: 1; }
.cover .team { font: 600 11.5pt/1.4 SERIF; letter-spacing: .14em; color: LILAC; margin: 0 0 16mm; }
.cover h1 { font: 600 32pt/1.18 SERIF; margin: 0 0 6mm; max-width: 12.5em; }
.cover .sub { font: 400 13.5pt/1.45 SERIF; color: LILAC; margin: 0 0 16mm; }
.cover dl { display: grid; grid-template-columns: 5.2em 1fr; row-gap: 1.8mm; margin: 0; font-size: 10pt; }
.cover dt { color: LILAC; }
.cover dd { margin: 0; }
.headline { margin-top: 16mm; display: grid; grid-template-columns: repeat(3, auto); justify-content: start;
  column-gap: 11mm; }
.headline div { border-left: 1.6pt solid RED; padding-left: 3.5mm; }
.headline b { display: block; font: 600 26pt/1.1 SERIF; white-space: nowrap; }
.headline span { font: 10pt/1.5 SERIF; }
.headline small { display: block; font: 8.2pt/1.4 SERIF; color: LILAC; }
.abstract { margin-top: auto; border-top: .6pt solid LILAC; padding-top: 6mm; }
.abstract h3 { font: 600 11pt SERIF; color: LILAC; margin: 0 0 2.5mm; letter-spacing: .14em; }
.abstract p { font: 10.6pt/1.7 SERIF; margin: 0; }
.spine { display: flex; flex-direction: column; align-items: center; justify-content: space-between;
  z-index: 1; }
.spine .name { writing-mode: vertical-rl; font: 600 22pt/1 KAI; color: LILAC; letter-spacing: .34em; }
.spine .roman { writing-mode: vertical-rl; font: 8.4pt SERIF; letter-spacing: .16em; margin-top: 5mm; }
.spine .names { display: flex; flex-direction: column; align-items: center; }
.stamp { text-align: center; font: 7.6pt/1.3 SERIF; color: LILAC; letter-spacing: .08em; }
.seal { width: 19mm; height: 19mm; background: RED; color: #fff; display: grid; margin: 0 auto 2.4mm;
  grid-template-columns: 1fr 1fr; direction: rtl; place-items: center; padding: 1.5mm;
  font: 600 17pt/1 KAI; border-radius: 1.4mm; }
h2 { display: grid; grid-template-columns: 12mm 1fr auto; align-items: center; margin: 9mm 0 3.5mm;
  padding-bottom: 2mm; border-bottom: 1.2pt solid PURPLE; break-after: avoid; }
h2 .num { width: 8.6mm; height: 8.6mm; background: RED; color: #fff; font: 600 13pt/8.6mm SERIF;
  text-align: center; border-radius: .9mm; }
h2 .en { font: 600 16pt/1.2 SERIF; color: PURPLE; }
h2 .zh { font: 400 11.5pt/1 KAI; color: PURPLE; opacity: .75; letter-spacing: .2em; }
blockquote { margin: 0 0 4mm; padding: .6mm 0 .6mm 4mm; border-left: 2pt solid RED; break-after: avoid; }
blockquote p { font: italic 11pt/1.6 SERIF; color: PURPLE; margin: 0; }
p, ul, ol { margin: 0 0 2.6mm; }
li { margin-bottom: 1mm; }
strong { color: PURPLE; }
a { color: PURPLE; }
table { border-collapse: collapse; width: 100%; margin: 3.5mm 0 5mm; font-size: 8.9pt; line-height: 1.45;
  border-top: 1.4pt solid PURPLE; border-bottom: 1.4pt solid PURPLE; }
tr { break-inside: avoid; }
th { text-align: left; font-weight: 600; color: PURPLE; padding: 1.7mm 2.4mm; border-bottom: .8pt solid RED; }
td { padding: 1.5mm 2.4mm; vertical-align: top; border-bottom: .3pt solid #dcd8e6; }
tr:last-child td { border-bottom: 0; }
code { font: 8.6pt Consolas, monospace; background: #F3F0F8; padding: 0 1.1mm; border-radius: .6mm; }
pre { background: #F3F0F8; border-left: 1.6pt solid PURPLE; padding: 2.4mm 3.5mm; break-inside: avoid; }
pre code { padding: 0; background: none; }
figure { margin: 5mm 0 6mm; break-inside: avoid; }
figure svg { display: block; width: 100%; height: auto; }
figcaption { margin-top: 2mm; font: 9pt/1.5 SERIF; }
figcaption b { color: PURPLE; margin-right: .5em; }
"""
COVER = """<section class="cover"><div class="lead">
<p class="team">{en} ({short})</p>
<h1>{title}</h1><p class="sub">{subtitle}</p>
<dl><dt>Team</dt><dd>{en} ({short}), {zh}</dd>
<dt>Event</dt><dd>More than a Hack 2027, Phase 1: Ball Possession</dd>
<dt>Version</dt><dd>pipeline {version}</dd></dl>
<div class="headline">{headline}</div>
<div class="abstract"><h3>Summary</h3><p>{abstract}</p></div></div>
<div class="spine"><div class="names"><div class="name">{zh}</div><div class="roman">{en}</div></div>
<div class="stamp"><div class="seal">{seal}</div>{seal_en}</div></div></section>"""


def text(x, y, content, size=11, fill=INK, anchor="start", weight=400):
    return ('<text x="{}" y="{}" font-size="{}" font-family=\'{}\' fill="{}" text-anchor="{}" '
            'font-weight="{}">{}</text>').format(x, y, size, SERIF, fill, anchor, weight, content)


def bar_chart(rows, top, fmt="{:.2f}"):
    left, width, step = 250, 330, 30
    parts = []
    for i, (label, value) in enumerate(rows):
        y = 16 + i * step
        length = width * value / top
        parts.append(text(left - 12, y + 9, label, anchor="end"))
        parts.append('<path d="M{0} {1} h{2} a4 4 0 0 1 4 4 v4 a4 4 0 0 1 -4 4 h-{2} z" fill="{3}"/>'.format(
            left, y, max(length - 4, 0), PURPLE))
        parts.append(text(left + length + 8, y + 9, fmt.format(value), weight=600))
    height = 16 + len(rows) * step
    parts.append('<line x1="{0}" y1="8" x2="{0}" y2="{1}" stroke="#9a94a8" stroke-width="1"/>'.format(left, height - 6))
    return '<svg viewBox="0 0 660 {}" role="img">{}</svg>'.format(height, "".join(parts))


def pipeline_figure():
    box, gap, parts = 98, 12, []
    for i, label in enumerate(STEPS):
        x = 2 + i * (box + gap)
        learned = i in (1, 3)
        parts.append('<rect x="{}" y="6" width="{}" height="40" rx="5" fill="{}" stroke="{}" '
                     'stroke-width="1.3"/>'.format(x, box, PURPLE if learned else "#fff", PURPLE))
        parts.append(text(x + box / 2, 30, label, 9.6, "#fff" if learned else INK, "middle", 600 if learned else 400))
        if i:
            parts.append('<path d="M{0} 26 h{1} m-4 -3.5 l4 3.5 l-4 3.5" fill="none" stroke="{2}" '
                         'stroke-width="1.2"/>'.format(x - gap + 1.5, gap - 3, PURPLE))
    return '<svg viewBox="0 0 660 52" role="img">{}</svg>'.format("".join(parts))


def figures(numbers):
    lengths = [(label + " s", value) for label, value in numbers["match_rate_by_length"]]
    return {
        "pipeline": (pipeline_figure(), "Figure 1.", "The pipeline. The learned models are filled; each is "
                     "followed by a decoder that turns per-frame probabilities into segments."),
        "scores": (bar_chart(numbers["scores"], 1.0), "Figure 2.", "spellF1 on the other-game matches, each "
                   "scored by models that never saw it."),
        "length": (bar_chart(lengths, 100, "{:g}%"), "Figure 3.", "Share of true possessions recovered at IoU "
                   "0.5, by possession length. Short possessions are where the remaining loss is."),
    }


def heading(match):
    number, zh, en = match.group(1), match.group(2).strip(), match.group(3).strip()
    return '<h2><span class="num">{}</span><span class="en">{}</span><span class="zh">{}</span></h2>'.format(
        number, en, zh)


def build_html(source, numbers):
    version = re.search(r"Pipeline version `([^`]+)`", source).group(1)
    body = markdown.markdown(source[source.index("## "):], extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r"<h2>(\d+)\.\s*([^|<]+)\|([^<]+)</h2>", heading, body)
    for key, (svg, label, caption) in figures(numbers).items():
        block = "<figure>{}<figcaption><b>{}</b>{}</figcaption></figure>".format(svg, label, caption)
        body = body.replace("<!-- figure:{} -->".format(key), block)
    css = CSS.replace("FOOTER", "{}   {}".format(TEAM_SHORT, TEAM_EN))
    for token, value in (("KAI", KAI), ("SERIF", SERIF), ("PURPLE", PURPLE), ("DEEP", DEEP), ("RED", RED),
                         ("INK", INK), ("LILAC", LILAC)):
        css = css.replace(token, value)
    headline = "".join("<div><b>{}</b><span>{}</span><small>{}</small></div>".format(*item)
                       for item in numbers["headline"])
    front = COVER.format(title=numbers["title"], subtitle=numbers["subtitle"], zh=TEAM_ZH, en=TEAM_EN,
                         short=TEAM_SHORT, version=version, abstract=numbers["abstract"], headline=headline,
                         seal="".join("<span>{}</span>".format(c) for c in SEAL), seal_en=SEAL_EN)
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><style>{}</style></head>"
            "<body>{}{}</body></html>").format(css, front, body)


def main():
    parser = argparse.ArgumentParser(description="technical document markdown -> A4 PDF via headless Edge")
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--numbers", default=os.path.join(HERE, "report_numbers.json"))
    args = parser.parse_args()

    with open(args.source, encoding="utf-8") as handle:
        source = handle.read()
    with open(args.numbers, encoding="utf-8") as handle:
        html = build_html(source, json.load(handle))
    browser = next(path for path in BROWSERS if os.path.exists(path))
    with tempfile.TemporaryDirectory() as work:
        page = os.path.join(work, "doc.html")
        with open(page, "w", encoding="utf-8") as handle:
            handle.write(html)
        subprocess.check_call([
            browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            "--user-data-dir=" + os.path.join(work, "profile"),
            "--print-to-pdf=" + os.path.abspath(args.target), "file:///" + page.replace("\\", "/"),
        ])
    print("wrote {} bytes".format(os.path.getsize(args.target)))


if __name__ == "__main__":
    main()
