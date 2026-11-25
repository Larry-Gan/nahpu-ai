# NEON label photos

Real specimen labels for the OCR test, from the NEON Biorepository. The photos
are fetched by `scripts/fetch-model.sh` into `models/`, not committed, and read
byte for byte as downloaded (no cropping or re-encoding). Each
`*.reference.json` here is what PaddleOCR 3.7.0 read on the photo of that name,
written by `scripts/ocr/make_references.py`.

**Licence:** CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/),
per each image record and the Biorepository's citation page,
https://biorepo.neonscience.org/portal/misc/cite.php.

## ocr08_mammal_sorex_trowbridgii_handwritten_notes.jpg

Whole shrew on a card with a loose handwritten label and a barcode sticker.

- NEON (National Ecological Observatory Network) Biorepository. Accessed via
  NEON Biorepository Sample Portal 2026-10-07. D00000028740-2_lg.jpg.
  Specimen image.
  https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/D00000028740-2_lg.jpg
- Catalogue number NEON01IU3 (https://doi.org/10.58052/NEON01IU3), *Sorex
  trowbridgii*, collection NEON-MAMC-VSS
- Record: https://biorepo.neonscience.org/portal/collections/individual/index.php?occid=406575

## ocr11_plant_hieracium_scouleri_herbarium_printed_label.jpg

Herbarium sheet with a small printed label, bottom right.

- NEON (National Ecological Observatory Network) Biorepository. Accessed via
  NEON Biorepository Sample Portal 2026-10-07. B00000053749.jpg. Specimen
  image.
  https://biorepo.neonscience.org/media/NEON_TPLC-HV/B00000053/B00000053749.jpg
- Catalogue number NEON02WPU (https://doi.org/10.58052/NEON02WPU), *Hieracium
  scouleri*, collection NEON-TPLC-HV
- Record: https://biorepo.neonscience.org/portal/collections/individual/index.php?occid=756498

## ocr01_beetle_pasimachus_sublaevis_label_closeup.jpg

Pinned ground beetle in front of its printed labels, which it partly covers.

- NEON (National Ecological Observatory Network) Biorepository. Accessed via
  NEON Biorepository Sample Portal 2026-10-07. NEON.BET.D03.001188_med.jpg.
  Specimen image.
  https://biorepo.neonscience.org/media/CMNH_INV/NEONBET/NEON.BET.D03.001188_med.jpg
- Catalogue number: none in the record; the label reads NEON.BET.D03.001188.
  *Pasimachus sublaevis*, collection CMNH-INV
- Record: https://biorepo.neonscience.org/portal/collections/individual/index.php?occid=762638

## ocr07_mammal_blarina_brevicauda_handwritten_tag.jpg

Whole shrew with a pre-printed NEON tag filled in by hand in block capitals.

- NEON (National Ecological Observatory Network) Biorepository. Accessed via
  NEON Biorepository Sample Portal 2026-10-07. B00000031028-1_1624657069.jpg.
  Specimen image.
  https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/B00000031028-1_1624657069.jpg
- Catalogue number NEON01IEC (https://doi.org/10.58052/NEON01IEC), *Blarina
  brevicauda*, collection NEON-MAMC-VSS
- Record: https://biorepo.neonscience.org/portal/collections/individual/index.php?occid=276607

## Photos without a test

Fetched with the others, as the examples behind the known limits in
`docs/setup.md`. Same source, licence and access date.

- `ocr02_beetle_pasimachus_sublaevis_label_sideways.jpg`: *Pasimachus
  sublaevis*, collection CMNH-INV, label NEON.BET.D03.001210.
  https://biorepo.neonscience.org/media/CMNH_INV/NEONBET/NEON.BET.D03.001210_med.jpg
- `ocr03_beetle_pasimachus_marginatus_label_occluded.jpg`: *Pasimachus
  marginatus*, collection CMNH-INV, label NEON.BET.D03.000206.
  https://biorepo.neonscience.org/media/CMNH_INV/NEONBET/NEON.BET.D03.000206_med.jpg
- `hard01_mammal_napaeozapus_skull_ink_number.jpg`: *Napaeozapus abietorum*,
  collection NEON-MAMC-VOS, https://doi.org/10.58052/NEON010U7
  https://biorepo.neonscience.org/media/NEON_MAMC-VOS/00000/skull_B00000061067-1_1648852819.jpg
- `hard02_mammal_sorex_rohweri_skull_ink_number.jpg`: *Sorex rohweri*,
  collection NEON-MAMC-VGB, https://doi.org/10.58052/NEON00XXK
  https://biorepo.neonscience.org/media/NEON_MAMC-VGBT/00000/D00000166461-10_1690845562.jpg
