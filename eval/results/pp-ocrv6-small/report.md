# OCR evaluation: PP-OCRv6 small

- Backend: burn-vulkan
- Compared with: PaddleOCR 3.7.0 on the same ONNX files, from `crates/nahpu_ocr_burn/tests/data/neon`, 4 photos
- Model load 0.77 s; per photo (decode + detect + recognise) median 1374 ms

The character error rate is the edit distance to PaddleOCR's text over its length, with our lines put in PaddleOCR's order. It says how closely this engine follows PaddleOCR, not how well either reads the label: there is no human transcription. PaddleOCR's lines scored below 0.5 are left out, with ours nearest to them.

| Photo | Size | Lines | PaddleOCR's lines | Character error rate | Left out (ours, PaddleOCR's) | Time |
|---|---|---|---|---|---|---|
| ocr01_beetle_pasimachus_sublaevis_label_closeup.jpg | 1400 x 935 | 8 | 8 | 1.1% | 0, 0 | 1783 ms |
| ocr07_mammal_blarina_brevicauda_handwritten_tag.jpg | 1400 x 933 | 18 | 17 | 1.5% | 4, 3 | 1091 ms |
| ocr08_mammal_sorex_trowbridgii_handwritten_notes.jpg | 1400 x 933 | 17 | 17 | 1.0% | 0, 0 | 1374 ms |
| ocr11_plant_hieracium_scouleri_herbarium_printed_label.jpg | 1400 x 2100 | 19 | 20 | 1.1% | 0, 0 | 1656 ms |

## ocr01_beetle_pasimachus_sublaevis_label_closeup.jpg

Read, in reading order:

```
D03.001188
FLO
A Polk County. Dis
WV
ess Preserve. 19m
75 Pitfal
t2013.RNelson
EON.DSNY.007
```

Compared lines that differ:

| PaddleOCR | Ours |
|---|---|
| W | WV |

## ocr07_mammal_blarina_brevicauda_handwritten_tag.jpg

Read, in reading order:

```
NEON01IEC
Ueu
COLLECTOR:D. MIKROS
SPECIES: BLARINA BREVICAMDA
TAGID:HARV.20181002.0010E(20181002.V SEX:
SITE:HARV
PLOT/TRAPID: 01O/E/
DATE:D 20181002
neon
Type ll barcode
B00000031028
METRIC
1011121314
115
Inlulmlmsulmlmlmiludmlm
iullul
Iμlmlmlmuulmlul
INCHES
```

Compared lines that differ:

| PaddleOCR | Ours |
|---|---|
| COLLECTOR:D.MIKROS | COLLECTOR:D. MIKROS |
| TAGID:HARV.2018100Z.0010E(20181002.VSEX: | TAGID:HARV.20181002.0010E(20181002.V SEX: |

## ocr08_mammal_sorex_trowbridgii_handwritten_notes.jpg

Read, in reading order:

```
sex:v
NEON011U3
nen Type IV barcode
Specics:SoREk trowbridgii
WREF.20190930.0001A820190930.√
D00000028740
collccted as part of the Nationnl Ecological
observntosy
'Network
METRIC 1
6
8
110
13
114
15
INCCHES
```

Compared lines that differ:

| PaddleOCR | Ours |
|---|---|
| collected as part of the National Ecological | collccted as part of the Nationnl Ecological |

## ocr11_plant_hieracium_scouleri_herbarium_printed_label.jpg

Read, in reading order:

```
9
10
ASU
cm
copyright reserved
Type Il barcode
B00000053749
Plants of NEON Domain D16 — Pacific Northwest
Asteraceae
Hieraċium scouleri Hook.
United States, Washington, Clark County. Abby Road Site,
45.7451 N -122.3830 W. WGS84. 480.26 m.
Gravel road within young forest stand.With Pseudotsuga
menziesii var. menziesii, Hypochaeris radicata, Polystichum
munitum.
H.Spencer
014
2019-06-19
Collected as part of the National Ecological Observatory Network
```

Compared lines that differ:

| PaddleOCR | Ours |
|---|---|
| neen | (nothing) |
