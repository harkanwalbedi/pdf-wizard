# PDF Wizard evals

The eval set measures how accurately PDF Wizard fills forms with a live Gemini
model. Use it to pick the model and the confidence threshold, and rerun it after
every prompt or model change: no change should ship if precision drops.

## Run

```bash
export GOOGLE_CLOUD_PROJECT=your-project   # Vertex AI, as for `pdf-wizard fill`
gcloud auth application-default login

.venv/bin/python -m evals.run                                    # default model, thresholds 0.5-0.9
.venv/bin/python -m evals.run --model gemini-3.8-flash --model gemini-2.5-pro --repeat 3
.venv/bin/python -m evals.run --case irs_w9 --show-values --out evals/results/latest.json
```

> [!WARNING]
> Each run sends every case to every model: cases × models × repeats paid model
> calls. Trying more thresholds is free, because each model answer is replayed
> for every threshold.

The runner prints one row per model and threshold and recommends the config
with the best recall among those reaching `--target-precision` (default 98%).
Ties go to the model listed first, so list the cheaper model first.

Exit codes: 0 = a config met the target, 1 = none did, 2 = a malformed case,
4 = configuration error.

## Metrics

Every writable field of each form is scored, using values read back from the
filled PDF:

| Outcome | Meaning |
|---|---|
| correct | filled with an accepted value |
| wrong | filled, but not with an accepted value |
| hallucinated | filled, but the field should have stayed blank |
| missed | left blank, but a value was expected |
| correct_blank | left blank, as expected |

- **precision** = correct / all filled fields
- **recall** = correct / fields that should be filled
- **hallucination rate** = hallucinated / fields that should stay blank

## Benchmark Cases

PDF Wizard includes **8 checked-in benchmark cases** across diverse layouts and source modalities:

### 1. Synthetic Stress Cases (`evals/make_cases.py`)

| Case | What it tests | Source Modality |
|---|---|---|
| `job_application` | Multi-type form: dates, 9-digit comb SSN, combobox, radio buttons, today's date, and blank fields | JSON record (`applicant.json`) + note |
| `w9_from_email` | Cryptic field names (`f1_01`), quoted distractor address, SSN split across 3 separate boxes | Email (`.eml`) |
| `intake_from_reference` | Clinic intake transfer: splitting combined lines, reformatting dates, avoiding guessing unstated fields (e.g. sex) | Previously filled PDF (`reference`) |
| `benefits_injection` | Prompt-injection resistance: malicious instructions embedded in an HR memo are rejected | Plain text (`hr_letter.txt`) |

### 2. Official IRS Government Form Cases

Real government AcroForms tested against realistic unstructured, scanned, and visual documents:

| Case | Form | What it tests | Source Modality |
|---|---|---|---|
| `irs_w9` | Form W-9 (Rev. 03-2024) | Business name, sole proprietor checkbox, requester address, 3-part comb SSN | **Scanned PDF** (`w9_contractor_scan.pdf` — image-only scan with zero text layer) |
| `irs_8822` | Form 8822 (Rev. 11-2021) | Dual taxpayer names/SSNs, prior vs new mailing addresses, telephone number | **JPEG Image** (`f8822_move_notice.jpg` — smartphone photo / flat scan) |
| `irs_w8ben` | Form W-8BEN (Rev. 10-2021) | International treaty claim, article/rate extraction, foreign TIN, international address | **PNG Image** (`w8ben_beneficial_owner.png` — high-contrast digital scan) |
| `irs_w4` | Form W-4 (2026) | Marital status mutex checkboxes, Step 2c two jobs box, child/dependent credits, employer EIN | **Unstructured Text + PDF** (`w4_employee.txt` + `w4_employer_memo.pdf`) |

```
evals/cases/<name>/
    form.pdf        the blank target form
    case.json       sources and options
    expected.json   field name -> expected value(s) or null
    sources/        files referenced by case.json
```

`expected.json` maps field names (see `pdf-wizard inspect form.pdf`) to:
- `null`: the field must stay blank;
- a string, or a list of acceptable strings (for format variants such as phone numbers or currency);
- `true`/`false` for checkboxes (`"checked"`/`"unchecked"` also work).

## Synthetic Dataset & Fake PDF Generator (`evals/dataset.py`)

To evaluate models at scale (10s or 100s of cases) or test a custom AcroForm before production, use the synthetic dataset generator. Powered by `Faker`, it synthesizes realistic ground-truth records, fills the target PDF via `fill_pdf`, renders the unstructured reference source in any chosen modality, and emits ready-to-run eval cases:

```bash
# 1. Generate 100 fake filled PDFs and eval cases across all 5 forms and 6 source formats
.venv/bin/python -m evals.dataset --template all --count 100 --out /tmp/synth_100

# 2. Generate 200 fake W-9 PDFs with only scanned PDF and JPEG sources
.venv/bin/python -m evals.dataset --template irs_w9 --count 200 --formats scanned_pdf,jpeg --out /tmp/w9_scans

# 3. Generate 50 fake PDFs for an arbitrary custom AcroForm PDF (synthesized from its schema)
.venv/bin/python -m evals.dataset --form path/to/custom_form.pdf --count 50 --out /tmp/custom_dataset

# 4. Benchmark Gemini against the generated dataset
.venv/bin/python -m evals.run --cases /tmp/synth_100/cases --model gemini-3.8-flash
```

For every record `1..N`, the generator writes:
1. `<out>/filled/<id>.pdf` — Ground-truth filled PDF populated via `fill_pdf`.
2. `<out>/cases/<id>/sources/...` — Unstructured source rendered in the chosen modality (`txt`, `scanned_pdf`, `jpeg`, `png`, `pdf`, `eml`).
3. `<out>/cases/<id>/` — Complete eval case (`form.pdf`, `case.json`, `expected.json`) compatible with `evals.harness`.

> [!CAUTION]
> Real forms and documents often contain personal data. Keep production documents
> out of version control, and use `--show-values` only on a trusted terminal.
> Results files (`--out`) never contain values.
