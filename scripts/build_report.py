"""Build reports/TinyCD_reproduction_report.pdf (plain-language report). Run make_report_figures.py first."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "TinyCD_reproduction_report.pdf"
F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", F + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

NAVY, BLUE, LIGHT, GREYTXT = colors.HexColor("#1a365d"), colors.HexColor("#2b6cb0"), colors.HexColor("#ebf4ff"), colors.HexColor("#4a5568")
body = ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14.2, spaceAfter=6, alignment=TA_LEFT)
h1 = ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=NAVY, spaceBefore=4, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="DV-B", fontSize=11.2, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=4)
cap = ParagraphStyle("cap", fontName="DV-I", fontSize=8.2, leading=11, textColor=GREYTXT, spaceAfter=10)
small = ParagraphStyle("small", parent=body, fontSize=8.4, leading=12)
bul = ParagraphStyle("bul", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
title = ParagraphStyle("title", fontName="DV-B", fontSize=25, leading=30, textColor=NAVY, spaceAfter=6)
sub = ParagraphStyle("sub", fontName="DV", fontSize=12, leading=17, textColor=GREYTXT, spaceAfter=14)
cell = ParagraphStyle("cell", fontName="DV", fontSize=8.6, leading=11.5)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")
W = A4[0] - 4.2 * cm


def P(t, s=body): return Paragraph(t, s)
def B(items): return [Paragraph(i, bul, bulletText="•") for i in items]


def fig(name, caption, width=W):
    w, h = PILImage.open(FIG / name).size
    return KeepTogether([Image(str(FIG / name), width=width, height=width * h / w), P(caption, cap)])


def table(rows, widths, head=True, zebra=True, bold_last_col=False):
    data = [[Paragraph(str(c), cellb if (head and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def callout(text, color="#ebf8ff", border="#2b6cb0"):
    t = Table([[Paragraph(text, body)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)), ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(border)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def footer(c, d):
    c.saveState(); c.setFont("DV", 7.5); c.setFillColor(GREYTXT)
    c.drawString(2.1 * cm, 1.2 * cm, "TinyCD reproduction report - 18 Oct 2026")
    c.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"Page {d.page}")
    c.restoreState()


S = []
# ------------------------------------------------------------------ page 1: summary
S += [Spacer(1, 0.6 * cm), P("Finding the differences between two pictures", title),
      P("Reproducing the TinyCD model and checking its published score", sub),
      callout("<b>The short version.</b> We took a very small change-detection model called <b>TinyCD</b> (about 285,000 settings, a 1.3 MB file), "
              "ran the authors' own test program on their own test pictures, and got <b>F1 = 91.07</b>. The paper reports <b>91.05</b>. "
              "<b>So the published result was reproduced successfully.</b>"),
      Spacer(1, 6),
      callout("<b>One correction to a common reading of these numbers.</b> 91.07 is <i>not</i> \"way better\" than 91.05. The gap is 0.02 points. "
              "Repeating the test on a slightly different random draw of the same pictures moves the score by about &plusmn;0.4 points, "
              "so the two numbers are, for every practical purpose, <b>the same</b>. The honest claim is: <b>\"we matched the paper\"</b>, not \"we beat it\". "
              "Matching is exactly what a reproduction is supposed to show.", color="#fffaf0", border="#dd6b20"),
      Spacer(1, 8), P("At a glance", h2),
      table([["Question", "Answer"],
             ["What does the model do?", "Takes two photos of the same place (before / after) and paints the parts that changed."],
             ["What was tested?", "2,048 before/after aerial-photo pairs (the official LEVIR-CD test set), each 256 &times; 256 pixels."],
             ["Paper's score (F1)", "91.05"],
             ["Score we measured (F1)", "<b>91.07</b> &nbsp; (95% range 90.63 - 91.47)"],
             ["Reproduced?", "<b>Yes.</b> All four headline scores are within 0.07 points of the paper (page 4)."],
             ["Overall pixel accuracy", "99.11% (but this number is flattering - see page 3)"],
             ["Speed / size", "about 18-21 ms per pair on an old GTX 1080 Ti graphics card, about 55 MB of GPU memory, 1.28 MB model file"],
             ["Where it works", "Buildings that appeared or disappeared in aerial photos that line up with each other."],
             ["Where it does not", "Very tiny changes; a different dataset (SYSU-CD) where it found almost nothing; anything outside aerial buildings was not tested."],
             ["Can a client use it commercially?", "<b>Not without written permission</b> from the authors (code is research / non-commercial only)."]],
            [4.6 * cm, W - 4.6 * cm]),
      Spacer(1, 4),
      P("<b>What \"reproduced\" means here.</b> We re-ran the <i>released, already-trained</i> model on the official test pictures using the authors' own, unmodified test script. "
        "We did <b>not</b> train the model again from scratch, and we did <b>not</b> reproduce the paper's second dataset (WHU-CD) - the reasons are on page 9.", small),
      PageBreak()]

# ------------------------------------------------------------------ page 2: what is it
S += [P("1. What is TinyCD, in plain words?", h1),
      P("Imagine two satellite photos of the same neighbourhood taken a few years apart. A person can spot the new houses by flipping between the photos. "
        "<b>Change detection</b> asks a computer to do the same: given photo <b>A</b> (before) and photo <b>B</b> (after), "
        "produce a black-and-white picture where <b>white means \"this spot changed\"</b>."),
      P("A naive approach is to subtract one photo from the other, pixel by pixel. That fails in practice, because shadows, season, sun angle and image quality change "
        "<i>everything</i>, not just the buildings. A trained model learns what a \"real\" change (a new building) looks like and ignores the rest."),
      P("TinyCD is a trained model built to be <b>small</b>. The table shows how small it is."),
      table([["Property", "TinyCD"], ["Number of learned settings (parameters)", "285,128"], ["File size of the trained model", "1.28 MB"],
             ["Input", "Two colour pictures of the same size (it was trained on 256 &times; 256 tiles)"],
             ["Output", "A 'change probability' for every pixel, from 0 (no change) to 1 (change). Anything above 0.5 is painted white."],
             ["Trained on", "LEVIR-CD: building changes seen from the air"],
             ["Source", "Codegoni, Lombardi &amp; Ferrari (2022), published in Neural Computing and Applications; code at github.com/AndreaCodegoni/Tiny_model_4_CD"]],
            [6.2 * cm, W - 6.2 * cm]),
      Spacer(1, 8),
      fig("fig_example_pipeline.png", "Figure 1 - A real test pair (LEVIR-CD, file test_01777). The model sees only the first two pictures. The last picture is the human-drawn "
                                      "answer, which is used only to grade the model afterwards."),
      P("Notice how closely the white shapes in the model's mask follow the human-drawn answer. That is what a good score looks like."),
      P("Why we picked it", h2),
      P("The task was to find <b>one lightweight model</b> that takes two images and outputs a pixel-level change mask, with public code and a downloadable trained file, "
        "that can run on a modest graphics card. TinyCD fits all of those requirements. We searched for a better lightweight alternative and found none with public, "
        "ready-to-use weights that was clearly stronger, so we kept TinyCD. (We did not run a head-to-head test of other models.)"),
      PageBreak()]

# ------------------------------------------------------------------ page 3: evaluation
S += [P("2. How we measure the score (the 'evaluation matrix')", h1),
      P("To grade the model we compare its white-and-black mask against the human-drawn answer, <b>pixel by pixel</b>. Each pixel falls into one of four groups:"),
      table([["", "Model says <b>changed</b>", "Model says <b>not changed</b>"],
             ["Human says <b>changed</b>", "<b>True positive</b> - correct catch", "<b>False negative</b> - a miss"],
             ["Human says <b>not changed</b>", "<b>False positive</b> - a false alarm", "<b>True negative</b> - correctly left alone"]],
            [4.4 * cm, (W - 4.4 * cm) / 2, (W - 4.4 * cm) / 2]),
      Spacer(1, 6),
      P("From these four counts we get the scores used throughout the report:"),
      table([["Score", "Plain-English meaning", "Our value"],
             ["<b>Precision</b>", "Of everything the model painted white, how much was really a change? (low = many false alarms)", "92.65%"],
             ["<b>Recall</b>", "Of all the real changes, how much did the model find? (low = many misses)", "89.54%"],
             ["<b>F1</b>", "One number balancing precision and recall. <b>This is the headline score.</b>", "<b>91.07%</b>"],
             ["<b>IoU</b>", "How much the model's white area overlaps the human white area, divided by their combined area. A stricter score than F1.", "83.60%"],
             ["Accuracy", "Share of <i>all</i> pixels that are right. Misleading here (see below).", "99.11%"]],
            [2.6 * cm, W - 2.6 * cm - 2.4 * cm, 2.4 * cm]),
      Spacer(1, 8),
      callout("<b>Why accuracy alone is a trap.</b> In these photos only about 5% of the pixels are changes (6.8 million of 134 million). "
              "A lazy model that painted <i>nothing</i> white would still be about 95% 'accurate'. That is why change detection uses <b>F1</b> and <b>IoU</b>, "
              "which only reward the model for actually finding the changes."),
      Spacer(1, 6),
      fig("fig_error_map.png", "Figure 2 - Grading one pair. Green = correct, red = false alarm, blue = missed. Nearly everything is green; "
                                "the errors are thin edges around the buildings (8,816 correct pixels vs 304 false alarm and 940 missed).", width=W * 0.66),
      P("The settings we used", h2),
      *B(["<b>Cut-off:</b> a pixel counts as 'changed' when the model's probability is above 0.5. This is the authors' own setting. We did <b>not</b> adjust it using the test pictures.",
          "<b>Pooling:</b> the four counts are added up over all 2,048 test pairs before computing the scores, exactly as in the authors' test program.",
          "<b>Error bars:</b> we reshuffled the test pairs 1,000 times (resampling with replacement) to see how much the score would wobble. "
          "The F1 score stays between 90.63 and 91.47 in 95% of reshuffles. This is the 'give or take' to keep in mind when comparing 91.07 with 91.05."]),
      PageBreak()]

# ------------------------------------------------------------------ page 4: results
S += [P("3. The score we reproduced", h1),
      P("We ran the authors' unmodified <font name='DV-B'>test_ondata.py</font> with their released file <font name='DV-B'>levir_best.pth</font> on the official LEVIR-CD test pictures. Here is the comparison:"),
      table([["Score", "Paper", "Our run", "Difference", "Our 95% range"],
             ["Precision", "92.68", "92.65", "-0.03", "92.31 - 93.02"],
             ["Recall", "89.47", "89.54", "+0.07", "88.76 - 90.27"],
             ["<b>F1 (headline)</b>", "<b>91.05</b>", "<b>91.07</b>", "<b>+0.02</b>", "90.63 - 91.47"],
             ["IoU", "83.57", "83.60", "+0.03", "82.87 - 84.28"],
             ["Overall accuracy", "99.10", "99.11", "+0.01", "-"]],
            [3.4 * cm, 2.3 * cm, 2.3 * cm, 2.6 * cm, W - 10.6 * cm]),
      Paragraph("Paper values are from Table 6 / Table 7 of the published paper (arXiv 2207.13159). 2,048 test pairs, LEVIR-CD-256 split of 7,120 / 1,024 / 2,048 (train / validation / test).", cap),
      fig("fig_paper_vs_ours.png", "Figure 3 - Paper (grey) against our run (blue). The bars are the same height to within a hair. The black whiskers show the 95% give-or-take: "
                                   "the paper's value sits inside our whiskers for every score. The vertical axis starts at zero so the picture is not exaggerated.", width=W * 0.56),
      P("How do we know it is a genuine reproduction?", h2),
      *B(["<b>Official code, official file.</b> The model weights came from the authors' repository (checked with a fingerprint, SHA-256 &nbsp;743645b8&hellip;4a3e). The test program is the authors' own; the only change was a one-line compatibility fix for a newer version of Python, with no change to the maths.",
          "<b>Official data format.</b> 256 &times; 256 tiles, 2,048 test pairs - the numbers the paper states. Our image loading matches the official loader exactly (difference = 0.0).",
          "<b>Two independent runs agree.</b> The authors' script and our own evaluation script (which adds the error bars) produced the same F1 down to 16 decimal places: 0.9106975154790973.",
          "<b>No peeking.</b> We did not tune any setting on the test pictures. The 0.5 cut-off is the authors' value, and the sweep we ran on separate <i>validation</i> pictures found that 0.5 was already almost optimal (best 0.44, F1 90.85 vs 90.80 at 0.5)."]),
      P("So is 91.07 'better' than the paper's 91.05?", h2),
      P("No - they are the same result. A difference of 0.02 points is about <b>twenty times smaller</b> than the natural wobble (&plusmn;0.4) shown by the error bars. "
        "Differences this small come from ordinary things such as a different graphics card or library version. "
        "We would only claim an <i>improvement</i> if our whole range sat clearly above the paper's number; it does not. "
        "The right conclusion is that <b>the paper's claim holds up when somebody else re-runs it</b>."),
      PageBreak()]

# ------------------------------------------------------------------ page 5: examples and baseline
S += [P("4. What the results look like", h1),
      P("Beyond the headline score, we ran the pretrained model on a small, randomly chosen sample (seed 0) of <b>20 test pairs</b> (14 with changes and 6 labelled as 'nothing changed') "
        "and compared it with two simple methods that just subtract the colours of the two photos. These are the pictures from our notebook."),
      fig("nb_levir_changed.png", "Figure 4 - Six pairs with changes. Columns: before, after, human answer, model probability, model mask, and the simple colour-difference method. "
                                  "TinyCD (columns 4-5) tracks the human answer; the simple method (last column) lights up roads, trees and shadows. "
                                  "Row 3 shows a real miss: one small building was not found at all."),
      PageBreak(),
      fig("nb_levir_nochange.png", "Figure 5 - Four pairs where, according to the human labels, nothing changed. A good model leaves the mask black. "
                                   "TinyCD stays black on these; the simple method paints a large part of the picture white (false alarms)."),
      P("Comparison with the simple colour-difference methods", h2),
      table([["Method (same 20 pairs)", "Precision", "Recall", "F1", "IoU", "'No change' pairs with false alarm"],
             ["<b>TinyCD (pretrained)</b>", "0.926", "0.864", "<b>0.894</b>", "0.808", "<b>0 of 6</b>"],
             ["Colour difference, fixed cut-off (103)*", "0.081", "0.125", "0.099", "0.052", "6 of 6"],
             ["Colour difference, Otsu automatic cut-off", "0.064", "0.327", "0.107", "0.056", "6 of 6"]],
            [5.0 * cm, 2.0 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, W - 11.7 * cm]),
      Paragraph("*The fixed cut-off was chosen on the separate validation set (1,024 pairs), not on these pairs. With only 20 pairs these numbers are a sanity check, not a benchmark; "
                "the full 2,048-pair result is on page 4.", cap),
      fig("fig_baselines.png", "Figure 6 - On the same 20 pairs, TinyCD scores about nine times higher than simply subtracting the two photos.", width=W * 0.62),
      PageBreak()]

# ------------------------------------------------------------------ page 6: strengths and weaknesses
S += [P("5. What it does well and where it struggles", h1),
      P("Using all 2,048 test pairs, we also broke the score down by the size of the change. This answers the question \"does it catch small things?\""),
      fig("fig_by_size.png", "Figure 7 - Left: share of individual changed objects the model found, by object size (a changed object counts as found if at least half of it is covered). "
                             "Right: F1 score depending on how much of the picture changed. 1 pixel = 0.5 metres on the ground, so 50 pixels is roughly a garden shed.", width=W * 0.62),
      P("Reliable", h2),
      *B(["<b>Normal-sized buildings that appear or disappear.</b> 96-98% of objects larger than 200 pixels are found, with F1 above 90 when the change covers more than 5% of the picture.",
          "<b>Not crying wolf.</b> On the 1,113 test pairs that contain no change at all, only 33 (3.0%) had even a single false-alarm pixel, and only 0.011% of those pictures' pixels were wrongly painted white.",
          "<b>Ignoring colour and shadow changes</b> that fool the simple method (Figures 4 and 5)."]),
      P("Struggles", h2),
      *B(["<b>Very small changes.</b> Objects under 50 pixels are found only 20% of the time, and 50-200 pixels about 68%. When only 0-1% of the picture changes, F1 drops to about 60. "
          "<i>Trade-off:</i> the model prefers to miss a tiny change rather than raise a false alarm.",
          "<b>A different dataset (SYSU-CD).</b> Using the same model on 4,000 SYSU-CD pairs it had never seen, F1 collapsed to 0.003 and it found only 0.16% of the changes. "
          "Two reasons: the model was unsure on new imagery (its most confident pixel in a picture averaged only 0.14 out of 1), and SYSU-CD's human answers are rough shaded regions, not building outlines. "
          "This is a warning about moving to new data, not a fair score for the model (Figure 8)."]),
      PageBreak(),
      P("The cross-dataset test (SYSU-CD) in pictures", h2),
      P("SYSU-CD is a different collection of aerial photos (Hong Kong, 2007-2014). The model was never trained on it. We ran the LEVIR-trained model on all 4,000 SYSU-CD test pairs unchanged "
        "(F1 0.003; precision 82%, recall 0.16%): when it does paint something it is usually right, but it stays silent almost everywhere. "
        "A threshold sweep on SYSU validation pictures was <b>not</b> run, so we cannot say whether a lower cut-off would have helped."),
      fig("nb_sysu_cross.png", "Figure 8 - Cross-dataset test on SYSU-CD. The model does react to some new buildings (bright patches in column 4) but weakly, "
                               "and the human-drawn answers (column 3) are coarse blobs. On this data even the simple colour method scored higher (F1 0.45 vs 0.00 on a 10-pair sample).", width=W * 0.62),
      PageBreak()]

# ------------------------------------------------------------------ page 7: limits, licence, next
S += [P("6. Limits, licence and next steps", h1),
      P("What we did <u>not</u> do, and why", h2),
      *B(["<b>No retraining.</b> We reproduced the <i>test</i> result of the released model. Whether the paper's <i>training</i> recipe would give the same file again was not checked.",
          "<b>The second paper dataset (WHU-CD, paper F1 91.74) was not reproduced.</b> The official download page blocks automated downloads, and the authors' pre-processed copy on Google Drive returned an access-denied error (HTTP 401). "
          "The authors split WHU-CD <i>randomly</i>, so a split we made ourselves could contain tiles the model trained on, which would make the score meaningless. The WHU file (<font name='DV-B'>whu_best.pth</font>) is in the repository but was not tested.",
          "<b>No test on ordinary photos, products, text or screenshots.</b> The model was only trained and tested on aerial building changes. Nothing here shows it works elsewhere.",
          "<b>Not tested on misaligned or badly lit pairs.</b> The authors say it assumes the two photos line up. Also unknown: how the authors chose the 'best' saved model (validation or test pictures) - a small caveat on the published number."]),
      P("Licence - important for a client project", h2),
      callout("The TinyCD code and weights are released <b>\"for non-commercial and research purposes only\"</b>; there is no separate licence file and the authors ask commercial users to contact them. "
              "The LEVIR-CD pictures may be used for <b>academic purposes only</b>. SYSU-CD states no licence. <b>This work is therefore not cleared for a commercial client prototype</b> "
              "until the authors give written permission. Using it for an internal research demo is fine.", color="#fff5f5", border="#e03131"),
      P("Recommended next steps", h2),
      *B(["<b>Decide the client's real pictures.</b> If they are aerial or satellite photos of buildings, TinyCD is a sensible starting point. If they are product photos, documents or app screenshots, "
          "this model was not built for that and would need new labelled examples.",
          "<b>If performance is not enough:</b> collect a small set of the client's own labelled pairs (even 100-200), test the unchanged model on them, and only then consider fine-tuning.",
          "<b>If commercial use matters:</b> email the authors for permission, or pick a model with a permissive licence and repeat this same test.",
          "<b>Finish WHU-CD</b> once the authors' pre-processed copy (or a verified mirror) is available."]),
      P("Reproduce it yourself", h2),
      P("Project folder <font name='DV-B'>image difference/</font>: notebook <font name='DV-B'>notebooks/tinycd_demo.ipynb</font>, <font name='DV-B'>scripts/evaluate.py</font> (full test with error bars), "
        "<font name='DV-B'>results/</font> (logs and numbers quoted here; the official-script run is <font name='DV-B'>results/logs/official_test_levir_on_levir.log</font>) and the README."),
      table([["Setup used", ""], ["Computer", "NVIDIA GeForce GTX 1080 Ti (11 GB), Linux, Python 3.10.12"],
             ["Software", "PyTorch 2.7.1 (CUDA 11.8); official TinyCD code commit 1696827 (12 Jan 2024)"],
             ["Model file", "levir_best.pth, 1,276,499 bytes, SHA-256 743645b87199c96b&hellip;4a3e"],
             ["Data / timing", "LEVIR-CD 256 &times; 256 tiles from the Hugging Face mirror ericyu/LEVIRCD_Cropped256 (2,048 test pairs, matches the paper's split sizes); SYSU-CD from ericyu/SYSU_CD. 18.4 ms per pair (model only); whole 2,048-pair test in 32 s."]],
            [3.2 * cm, W - 3.2 * cm])]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                        title="Reproducing TinyCD: finding the differences between two pictures", author="")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print(OUT)
