"""Validate collected H2 records and prepare tables, prose and a matched figure.

Run with a Python environment containing Pillow. Does not change collected evidence.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE / "collected" / "experiment"
OUT = BASE / "assignment_ready"
OUT.mkdir(exist_ok=True)
SRC = BASE.parents[1] / "src" / "mfdime"


def read(p):
    return json.loads(p.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


wrapper = read(ROOT / "protocol.json")
protocol = wrapper["protocol"]
assert digest(protocol) == wrapper["signature"]
score = read(ROOT / "score" / "score.json")
assert score["experiment_signature"] == wrapper["signature"]
assert score == read(BASE / "score.json"), "Downloaded report differs from previously supplied report"
selection = read(BASE / "collected" / "selection.json")
arms = [f"{p}/{m}" for p, m in protocol["arms"]]
pair_names = {"340_339": "Zebra to sorrel", "293_286": "Cheetah to cougar", "285_283": "Egyptian cat to Persian cat"}
cohorts = {}
all_rows = {a: [] for a in arms}
for source, target in protocol["pairs"]:
    pair = f"{source}_{target}"
    cohort = read(ROOT / "cohorts" / f"{pair}.json")
    cohorts[pair] = cohort
    assert digest({k:v for k,v in cohort.items() if k != "cohort_sha256"}) == cohort["cohort_sha256"]
    assert cohort["signature"] == wrapper["signature"]
    assert not set(cohort["practice_names"]) & set(cohort["held_out_names"])
    held = zip(cohort["held_out_names"], cohort["check_predictions"], strict=True)
    assert cohort["names"] == [name for name, pred in held if pred == source]
    assert len(cohort["names"]) == score["pairs"][pair]["photos"]
    for seed in protocol["seeds"]:
        for arm in arms:
            folder = ROOT / "runs" / pair / f"seed{seed}" / arm
            rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()]
            marker = read(folder / "complete.json")
            assert marker["signature"] == wrapper["signature"]
            assert marker["cohort_sha256"] == cohort["cohort_sha256"]
            assert marker["rows_sha256"] == digest(rows)
            assert [r["photo"] for r in rows] == cohort["names"]
            for r in rows:
                assert r["seed"] == seed and r["source"] == source and r["target"] == target
                assert f'{r["peek"]}/{r["gradient_mode"]}' == arm
                sample = folder / "samples" / hashlib.sha256(r["photo"].encode()).hexdigest()[:20]
                sample_rows = [json.loads(line) for line in (sample / "manifest.jsonl").read_text().splitlines()]
                complete = read(sample / "complete.json")
                expected = {"signature":wrapper["signature"], "cohort_sha256":cohort["cohort_sha256"],
                            "source":source,"target":target,"photo":r["photo"],"seed":seed,
                            "peek":r["peek"],"gradient_mode":r["gradient_mode"]}
                assert complete["identity"] == expected and sample_rows == [r]
                assert complete["row_sha256"] == digest(r)
            all_rows[arm] += rows
for arm, rows in all_rows.items():
    reported = score["arms"][arm]
    assert len(rows) == reported["explanations"] == 355
    for field, metric in [("flipped", "guide_success"),("check_flipped", "check_success")]:
        assert abs(sum(r[field] for r in rows)/len(rows)-reported[metric]) < 1e-12
    assert sum(r["status"] == "numerical_failure" for r in rows) == reported["numerical_failures"] == 0
    for metric in ("seconds","pixel_l1","perceptual","generator_calls"):
        assert abs(sum(r[metric] for r in rows)/len(rows)-reported[metric]["mean"]) < 1e-8
sources = ("h2.py", "config.py", "dime.py", "flow.py", "explain.py", "setups.py", "dmf.py", "imagenet.py", "common.py")
h = hashlib.sha256()
for name in sources:
    h.update(name.encode()); h.update((SRC / name).read_bytes())
source_match = h.hexdigest() == protocol["code_sha256"]
assert source_match, "The repository's generation source differs from the protocol hash"

table = ["| Method | Guide target success | Evaluator target success | Mean seconds |", "|---|---:|---:|---:|"]
tex = [r"\begin{tabular}{lrrr}",r"\hline",r"Method & Guide success (\%) & Evaluator success (\%) & Time (s) \\",r"\hline"]
labels = ["1 jump, shortcut","1 jump, exact","4 jumps, shortcut","4 jumps, exact"]
for arm,label in zip(arms,labels,strict=True):
    a=score["arms"][arm]
    table.append(f'| {label} | {100*a["guide_success"]:.1f}% | {100*a["check_success"]:.1f}% | {a["seconds"]["mean"]:.2f} |')
    tex.append(f'{label} & {100*a["guide_success"]:.1f} & {100*a["check_success"]:.1f} & {a["seconds"]["mean"]:.2f} ' + r'\\')
tex += [r"\hline",r"\end{tabular}","",r"% Each arm: 71 photos x 5 paired seeds = 355 outcomes."]
tex += ["",r"\begin{tabular}{lrr}",r"\hline",r"Exact minus shortcut & Difference (pp) & 95\% interval (pp) \\",r"\hline"]
comparison = ["| Exact minus shortcut | Difference (pp) | Paired 95% interval (pp) |", "|---|---:|---:|"]
for peek,c in score["comparisons"].items():
    lo,hi=c["ci95"]
    comparison.append(f'| {peek} | {100*c["difference_exact_minus_shortcut"]:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    tex.append(f'{peek} & {100*c["difference_exact_minus_shortcut"]:+.2f} & [{100*lo:+.2f}, {100*hi:+.2f}] ' + r'\\')
tex += [r"\hline",r"\end{tabular}"]
(OUT / "tables.tex").write_text("\n".join(tex)+"\n")

# Assemble a scientific image panel without changing image content.
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(n):
    return ImageFont.truetype(font_path,n)


W, HEADER, ROW, LEFT, TILE, GAP = 1672, 105, 336, 256, 256, 20
canvas=Image.new("RGB",(W,HEADER+6*ROW+50),"white")
d=ImageDraw.Draw(canvas)
d.text((24,15),"Matched ImageNet counterfactual examples",font=font(26),fill="#17212b")
for j,label in enumerate(["Original","1 jump / shortcut","1 jump / exact","4 jumps / shortcut","4 jumps / exact"]):
    d.text((LEFT+j*(TILE+GAP),64),label,font=font(19),fill="#17212b")
example_lines=["# Figure examples", "", "Selection: first two photos in each frozen cohort; seed 0; all four arms.", "",
               "G = guide predicts target; E = independent evaluator predicts target. Other predictions are recorded as class IDs.",
               ""]
for i,ex in enumerate(selection["examples"]):
    pair=ex["pair"]; y=HEADER+i*ROW
    title=pair_names[pair]
    if pair=="285_283":
        title="Egyptian cat to\nPersian cat"
    d.multiline_text((24,y+12),title,font=font(20),fill="#17212b",spacing=5)
    photo_id=ex["photo"].split("_val_")[1].split("_")[0]
    d.text((24,y+86),f"Photo {photo_id}",font=font(17),fill="#455563")
    d.text((24,y+110),"Seed 0",font=font(17),fill="#455563")
    example_lines += [f'## {pair_names[pair]} — {ex["photo"]}', "",
                      "| Method | Guide class | Evaluator class | Guide target? | Evaluator target? |", "|---|---:|---:|---|---|"]
    origs=[]
    for j,arm in enumerate(arms):
        sample=ROOT/"runs"/pair/"seed0"/arm/"samples"/ex["sample_hash"]
        r=json.loads((sample/"manifest.jsonl").read_text().strip())
        imgs=sample/"images"
        orig=next(imgs.glob("*_orig.png")); cf=next(imgs.glob("*_cf.png"))
        origs.append(hashlib.sha256(orig.read_bytes()).hexdigest())
        for col,path in ([(0,orig),(j+1,cf)] if j==0 else [(j+1,cf)]):
            im=Image.open(path).convert("RGB")
            assert im.size == (256,256)
            canvas.paste(im,(LEFT+col*(TILE+GAP),y))
        gx="yes" if r["flipped"] else "no"; exx="yes" if r["check_flipped"] else "no"
        x=LEFT+(j+1)*(TILE+GAP)
        d.text((x,y+266),f"G target: {gx}   E target: {exx}",font=font(17),fill="#17212b")
        d.text((x,y+292),f'Classes: G {r["guide_after"]}, E {r["check_after"]}',font=font(15),fill="#455563")
        example_lines.append(f'| {arm} | {r["guide_after"]} | {r["check_after"]} | {gx} | {exx} |')
    assert len(set(origs))==1, "Original images differ across matched arms"
    d.text((LEFT,y+266),"Both classify as source",font=font(17),fill="#17212b")
    example_lines += [""]
d.text((24,HEADER+6*ROW+5),"G: ResNet-50 guide. E: ViT-B/16 evaluator. Selection is independent of generated outcomes.",
       font=font(19),fill="#455563")
canvas.save(OUT/"matched_examples.png",dpi=(220,220))
for pair,index in [("340_339",0),("293_286",2),("285_283",4)]:
    # A smaller two-row alternative for assignments with limited page width/space.
    crop=canvas.crop((0,HEADER+index*ROW,W,HEADER+(index+2)*ROW))
    panel=Image.new("RGB",(W,HEADER+2*ROW+50),"white")
    panel.paste(canvas.crop((0,0,W,HEADER)),(0,0)); panel.paste(crop,(0,HEADER))
    panel.paste(canvas.crop((0,HEADER+6*ROW,W,canvas.height)),(0,HEADER+2*ROW))
    panel.save(OUT/f"examples_{pair}.png",dpi=(220,220))
(OUT/"example_predictions.md").write_text("\n".join(example_lines))

cohort_table=["| Pair | Validation candidates | Guide correct | Reserved practice | Held-out candidates | Final eligible |",
              "|---|---:|---:|---:|---:|---:|"]
for pair,c in cohorts.items():
    cohort_table.append(f'| {pair_names[pair]} | {c["candidates"]} | {c["guide_correct"]} | {len(c["practice_names"])} | '
                        f'{len(c["held_out_names"])} | {len(c["names"])} |')
pair_table=["| Pair | Photos | One-jump difference (pp), 95% interval | Four-jump difference (pp), 95% interval |",
            "|---|---:|---:|---:|"]
for pair,p in score["pairs"].items():
    cells=[]
    for peek in ("jump1","jump4"):
        c=p["comparisons"][peek]; lo,hi=c["ci95"]
        cells.append(f'{100*c["difference_exact_minus_shortcut"]:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]')
    pair_table.append(f'| {pair_names[pair]} | {p["photos"]} | {cells[0]} | {cells[1]} |')
text='''# ImageNet H2: assignment material

## Methods draft

We compared shortcut and exact differentiation through one- and four-jump flow-map look-aheads on three fixed ImageNet
class pairs: zebra to sorrel, cheetah to cougar, and Egyptian cat to Persian cat. ResNet-50 guided the edits; a fixed
ViT-B/16 independently evaluated target-class success and did not guide generation or retries. The first 16
guide-correct images per pair were reserved for practice. The remaining guide-correct images were filtered for correct
source classification by the evaluator, leaving 71 held-out originals (31, 30, and 10 per pair). Every eligible
original was processed with seeds 0–4 in all four arms, yielding 355 outcomes per arm and 1,420 total outcomes.

All arms used batch size one, gain 1, the 60-step guidance schedule, scales 8/10/15 with the existing retry policy,
and shared photo/seed noise streams across arms. No gradient normalization or update-norm matching was applied.
Experiments ran on NVIDIA A100-PCIE-40GB GPUs with PyTorch 2.8.0+cu126 and TF32 enabled. Runtime includes guidance,
final decoding, and diagnostic logging, and excludes evaluator scoring and setup.

The primary endpoint was evaluator target-class success over all eligible originals, including guide failures.
Exact-minus-shortcut differences were estimated separately for each jump count. The 95% percentile intervals used
5,000 paired bootstrap draws, resampling photos within class pair while retaining all seeds and arms together; pooled
estimates weighted each photo equally. Intervals are reported separately for the two prespecified comparisons without
a multiplicity adjustment. Seeds are repeated outcomes for a photo, not additional independent photos.

## Results draft

Exact differentiation through the one-jump look-ahead increased evaluator target success from 9.9% (35/355) to 15.2%
(54/355): a gain of 5.35 percentage points (paired 95% bootstrap interval: 2.25 to 8.73). Guide target success
decreased from 96.9% to 94.6%, while mean runtime increased from 6.73 to 8.59 seconds per explanation (27.5%). With
four jumps, evaluator target success increased from 10.4% (37/355) to 12.7% (45/355), but the difference remained
uncertain: 2.25 percentage points (95% interval: -1.69 to 6.20). No numerical failures were recorded in any arm.

'''+"\n".join(table)+"\n\n"+"\n".join(comparison)+'''

Table caption: Results on 71 held-out originals and five paired seeds. Guide and evaluator success are percentages of
all 355 outcomes per arm. Differences are exact minus shortcut in percentage points. Photo-level resampling preserves
seed and arm pairing.

## Discussion draft

The results support the one-jump H2 comparison under this fixed protocol, while the four-jump comparison remains
inconclusive. One-jump gains were positive in all three class-pair point estimates, although uncertainty varied by
pair. Exact differentiation also increased mean pixel L1 change (0.233 to 0.242) and the recorded perceptual distance
(381.1 to 401.3). The experiment therefore measures the combined effects of gradient magnitude and direction under
common settings; it does not isolate direction at matched edit size.

Absolute transfer success remained modest despite high guide success. Independent-classifier transfer is a proxy for
semantic validity, and these results do not establish improved human-perceived realism. Generalization is limited by
the three class pairs and the small cat cohort of ten photos. The bootstrap describes photo-level uncertainty within
these fixed pairs, not uncertainty over arbitrary ImageNet class pairs or new seed draws. A clear interval for one
jump and an inconclusive interval for four jumps do not establish an interaction between jump count and gradient mode.
The observed one-jump exact success rate exceeds four-jump exact, but this cross-jump contrast was not tested here.

Inspection of the six fixed examples shows substantial artifacts in some outputs, including distorted animal anatomy
and altered or removed faces. Such artifacts also occur in outputs for which both classifiers predict the target. This
small illustrative panel reinforces the need to distinguish classifier transfer from realistic counterfactual
transformation; it does not estimate the frequency of visual artifacts across the full cohort.

## Figure and caption

![Matched examples](matched_examples.png)

Figure caption: Matched original and counterfactual images for the first two photos in each frozen held-out cohort,
using seed 0. Columns show the original, one-jump shortcut, one-jump exact, four-jump shortcut, and four-jump exact
outputs. G and E denote the ResNet-50 guide and ViT-B/16 evaluator; target labels indicate whether each predicts the
target class. Examples were selected independently of generated outcomes and illustrate outputs rather than constitute
a visual-quality evaluation. Actual predicted class IDs are included below each edited image. Predictions were
recorded before PNG quantization.

For smaller figures, use the separate two-row `examples_<source>_<target>.png` panels. Exact photo identities and
predictions are in `example_predictions.md`.

## Appendix: cohort accounting

'''+"\n".join(cohort_table)+"\n\n## Appendix: pair-specific estimates\n\n"+"\n".join(pair_table)+'''

## Evidence checks

All 60 arm manifests and 1,420 sample records were checked for identity, ordering, completion hashes, and agreement.
Counts, guide/evaluator success rates, runtime, image distances, and generator-call means agree with the score report.
Frozen cohort hashes and practice/held-out separation were checked. The repository's nine generation source files
reproduce the protocol's code hash. Selected originals match across arms. The complete scorer was not rerun because
the compact archive intentionally omits unselected images; confidence intervals above come from the cluster report.
The manifest-based means were checked, not independently recomputed from image pixels.

Original evidence: `../collected/experiment/` and `../collected/logs/`. Keep the complete cluster outputs as the full archive.
'''
(OUT/"assignment_notes.md").write_text(text)
print(f"Validated {sum(len(v) for v in all_rows.values())} sample records; generation source hash matches.")
print(f"Prepared assignment material in {OUT}")
